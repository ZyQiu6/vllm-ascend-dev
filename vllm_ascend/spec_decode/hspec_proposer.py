# Copyright 2026 Xuyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
HSpec Proposer – on-device hidden-state similarity speculative decoding.

"""

import logging
import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType
from vllm_ascend.spec_decode.hspec_table import (
    GlobalHSpecTableGroup,
    get_hspec_tables,
)
from vllm_ascend.spec_decode.hspec_utils import prompt_id_from_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("HSPEC_LOG_LEVEL", os.getenv("VERL_LOGGING_LEVEL", "WARN")))


# Worker-local cached prompt table (on-device tensors + CPU refs)

class _CachedPromptTable:
    """Per-prompt table data cached on the worker's compute device.

    On-device tensors (mean, components, keys) enable hot-loop queries
    without any RPC or device↔host sync.  Value retrieval (draft tokens)
    is a sub-microsecond CPU numpy slice.
    """

    __slots__ = (
        "mean", "components", "keys",
        "rollout_seqs", "entry_rollout_idx", "entry_offset",
        "n_entries",
        "wnd_size", "max_wnd", "min_wnd",
    )

    def __init__(
        self,
        mean: torch.Tensor,
        components: torch.Tensor,
        keys: torch.Tensor,
        rollout_seqs: list,
        entry_rollout_idx: np.ndarray,
        entry_offset: np.ndarray,
        n_entries: int,
        wnd_size: int = 8,
        max_wnd: int = 28,
        min_wnd: int = 2,
    ):
        self.mean = mean                            # (D,)  float32, device
        self.components = components                  # (K,D) float32, device
        self.keys = keys                              # (M,K) float32, device, L2-norm'd
        self.rollout_seqs = rollout_seqs              # list[np.ndarray int32], CPU
        self.entry_rollout_idx = entry_rollout_idx    # (M,) int32, CPU
        self.entry_offset = entry_offset              # (M,) int32, CPU
        self.n_entries = n_entries
        self.wnd_size = wnd_size
        self.max_wnd = max_wnd
        self.min_wnd = min_wnd

    def get_draft_tokens(self, entry_idx: int, max_tokens: int) -> List[int]:
        """O(1) slice into the rollout token buffer."""
        if entry_idx < 0 or entry_idx >= self.n_entries:
            return []
        ridx = int(self.entry_rollout_idx[entry_idx])
        off = int(self.entry_offset[entry_idx])
        seq = self.rollout_seqs[ridx]
        return seq[off: off + max_tokens].tolist()

    def update_window(self, accept_length: int):
        """Congestion-control style adaptive window"""
        if accept_length >= self.wnd_size:
            self.wnd_size = min(self.wnd_size + 1, self.max_wnd)
        elif accept_length <= 1:
            self.wnd_size = max(self.wnd_size // 2, self.min_wnd)


# Main proposer

class HSpecProposer(Proposer):
    """HSpec Proposer with worker-local cache and on-device query.

    Hot-loop cost per decode step:
      • Per request: 1 matmul ``(1,D)×(D,K)``, 1 matvec ``(M,K)×(K,)``,
        1 argmax on device.  All O(DK + MK) on NPU.
      • Batch-wide: 1 ``torch.stack().cpu()`` for ``(P, 2)`` scalars
        (best_sim + best_idx) – negligible.
      • **Zero** Ray / ZMQ / network calls.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner,
    ):
        self.name = SpecDcodeType.HSPEC
        self.device = device
        self.runner = runner

        spec_config = vllm_config.speculative_config
        self.max_draft_tokens: int = spec_config.num_speculative_tokens
        self.similarity_threshold: float = getattr(
            spec_config, "hspec_similarity_threshold", 0.9,
        )
        self.min_match_len: int = getattr(
            spec_config, "hspec_min_match_len", 1,
        )

        # Distributed table manager (for prefetch *only*; never in hot loop)
        self.hspec_tables: GlobalHSpecTableGroup = get_hspec_tables(
            similarity_threshold=self.similarity_threshold,
        )

        # Worker-local cache
        self._cache: OrderedDict[str, _CachedPromptTable] = OrderedDict()
        self._not_in_table: Set[str] = set()   # prompts known absent
        self._cache_version: int = -1
        self._max_cache_size: int = 512

        # Async prefetch state (never blocking in hot loop)
        # Each entry: (ray.ObjectRef, [prompt_ids]) where the future
        # resolves to (version, {pid: table_data | None}).
        self._pending_fetches: List[tuple] = []
        self._pending_pids: Set[str] = set()

        # Accept-length tracking (adaptive window control)
        self._accept_lengths: Dict[str, int] = {}

        # Lightweight local metrics (for functional + perf validation).
        # These live in the vLLM worker process; we never RPC in the hot loop.
        self._stat_calls = 0
        self._stat_queries = 0
        self._stat_hits = 0
        self._stat_total_draft_len = 0
        self._stat_prefetch_fired = 0
        self._stat_prefetch_ready = 0
        self._stat_accept_sum = 0
        self._stat_accept_count = 0
        self._last_log_t = time.time()
        self._log_every_calls = int(os.environ.get("HSPEC_LOG_EVERY_CALLS", "200"))
        self._log_every_s = float(os.environ.get("HSPEC_LOG_EVERY_S", "10"))

        logger.info(
            "HSpec proposer initialised: threshold=%.3f, "
            "max_draft=%d, cache_cap=%d",
            self.similarity_threshold,
            self.max_draft_tokens,
            self._max_cache_size,
        )

    # async prefetch

    def prefetch_for_batch(self, req_ids: List[str]) -> None:
        """Fire async prefetch for a batch of requests – **non-blocking**.

        Called by ``model_runner`` **before** the forward pass so the
        Ray futures have the entire forward-pass latency (10-100 ms) to
        resolve.  By the time ``generate_token_ids()`` is invoked the
        cache is typically warm already.

        If a prompt is not ready yet, ``generate_token_ids()`` simply
        returns ``draft=[]`` for that request (graceful degradation).
        """
        # Convert req_ids → prompt_ids
        prompt_ids: List[str] = []
        for req_id in req_ids:
            req_state = self.runner.requests.get(req_id)
            if req_state is not None:
                prompt_ids.append(
                    prompt_id_from_token_ids(req_state.prompt_token_ids),
                )
        if not prompt_ids:
            return

        # Consume any futures that became ready since last call
        self._poll_pending()
        # Fire new async fetches for cache misses
        self._fire_prefetch_async(prompt_ids)

    def _poll_pending(self) -> None:
        """Non-blocking: consume any ready prefetch futures.

        Uses ``ray.wait(timeout=0)`` which returns immediately with
        whatever futures are already completed – typically ~1 µs Python
        overhead, zero network I/O.
        """
        if not self._pending_fetches:
            return

        import ray as _ray

        all_futures = [f for f, _ in self._pending_fetches]
        ready_refs, _ = _ray.wait(
            all_futures, num_returns=len(all_futures), timeout=0,
        )
        if not ready_refs:
            return
        ready_set = set(ready_refs)

        version_bumped = False
        still_pending: List[tuple] = []
        for future, pids in self._pending_fetches:
            if future not in ready_set:
                still_pending.append((future, pids))
                continue

            # Consume this ready future
            try:
                version, table_data = _ray.get(future)
                self._stat_prefetch_ready += 1

                if version < self._cache_version:
                    # Stale data from a previous epoch – discard silently
                    pass
                else:
                    if version > self._cache_version:
                        # Epoch swap detected → invalidate old cache
                        self._cache.clear()
                        self._not_in_table.clear()
                        self._cache_version = version
                        version_bumped = True

                    # Populate cache with fresh data
                    for pid in pids:
                        data = table_data.get(pid)
                        if data is not None:
                            try:
                                cached = self._build_cached_table(data)
                                self._cache[pid] = cached
                                self._cache.move_to_end(pid)
                            except Exception:
                                self._not_in_table.add(pid)
                        else:
                            self._not_in_table.add(pid)
            except Exception:
                # On error mark prompts as absent to avoid infinite retry
                for pid in pids:
                    self._not_in_table.add(pid)

            # Remove consumed pids from pending set
            for pid in pids:
                self._pending_pids.discard(pid)

        self._pending_fetches = still_pending

        # On epoch swap, abandon remaining (likely stale) pending
        # futures.  Their Ray ObjectRefs are GC'd harmlessly.  Fresh
        # fetches will be fired by the next _fire_prefetch_async() call.
        if version_bumped and self._pending_fetches:
            self._pending_fetches = []
            self._pending_pids.clear()

        # LRU eviction
        while len(self._cache) > self._max_cache_size:
            self._cache.popitem(last=False)

    def _fire_prefetch_async(self, prompt_ids: List[str]) -> None:
        """Fire async Ray futures for uncached prompts – **non-blocking**.

        Futures are appended to ``_pending_fetches`` and polled later
        by ``_poll_pending()``.  Prompts already cached, pending, or
        known-absent are skipped.
        """
        missing = [
            pid for pid in set(prompt_ids)
            if (pid not in self._cache
                and pid not in self._not_in_table
                and pid not in self._pending_pids)
        ]
        if not missing:
            return

        try:
            new_futures = self.hspec_tables.prefetch_batch_async(missing)
            for future, pids in new_futures:
                self._pending_fetches.append((future, pids))
                self._pending_pids.update(pids)
            if new_futures:
                self._stat_prefetch_fired += len(new_futures)
        except Exception:
            logger.debug("HSpec: async prefetch fire failed", exc_info=True)

    def _maybe_log_metrics(self) -> None:
        """Best-effort periodic metrics log (no blocking, minimal overhead)."""
        self._stat_calls += 1
        now = time.time()
        if (self._stat_calls % self._log_every_calls != 0) and ((now - self._last_log_t) < self._log_every_s):
            return
        self._last_log_t = now

        q = max(int(self._stat_queries), 1)
        h = int(self._stat_hits)
        match_rate = float(h) / float(q)
        avg_draft = float(self._stat_total_draft_len) / float(max(h, 1))
        avg_accept = (
            float(self._stat_accept_sum) / float(self._stat_accept_count)
            if self._stat_accept_count > 0
            else 0.0
        )
        logger.info(
            "HSpec online metrics: queries=%d hits=%d match_rate=%.3f "
            "avg_draft_len=%.2f avg_accept_len=%.2f cache_size=%d "
            "pending=%d prefetch_fired=%d prefetch_ready=%d version=%d",
            int(self._stat_queries),
            h,
            match_rate,
            avg_draft,
            avg_accept,
            len(self._cache),
            len(self._pending_fetches),
            int(self._stat_prefetch_fired),
            int(self._stat_prefetch_ready),
            int(self._cache_version),
        )

    def _build_cached_table(self, data: dict) -> _CachedPromptTable:
        """Convert serialised table data dict → on-device cached table."""
        mean = torch.from_numpy(
            data["mean"].astype(np.float32, copy=False),
        ).to(self.device, non_blocking=True)

        components = torch.from_numpy(
            data["components"].astype(np.float32, copy=False),
        ).to(self.device, non_blocking=True)

        keys = torch.from_numpy(
            data["keys"].astype(np.float32, copy=False),
        ).to(self.device, non_blocking=True)

        # Ensure rollout_seqs are numpy arrays on CPU
        rollout_seqs = []
        for s in data["rollout_seqs"]:
            if isinstance(s, np.ndarray):
                rollout_seqs.append(s)
            else:
                rollout_seqs.append(np.asarray(s, dtype=np.int32))

        return _CachedPromptTable(
            mean=mean,
            components=components,
            keys=keys,
            rollout_seqs=rollout_seqs,
            entry_rollout_idx=np.asarray(
                data["entry_rollout_idx"], dtype=np.int32,
            ),
            entry_offset=np.asarray(data["entry_offset"], dtype=np.int32),
            n_entries=int(data["n_entries"]),
            wnd_size=int(data.get("wnd_size", 8)),
            max_wnd=int(data.get("max_wnd", 28)),
            min_wnd=int(data.get("min_wnd", 2)),
        )

    # anchor hidden-state extraction

    @staticmethod
    def _extract_anchor_hs(
        i: int,
        sample_hidden_states: torch.Tensor,
        valid_sampled_token_ids: List[List[int]],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
    ) -> Optional[torch.Tensor]:
        """Return the correct anchor hidden state for request *i*.

        Handles all spec-decode paths:
        - Non-spec decode: ``sample_hidden_states[i]`` (1-to-1).
        - Spec, no drafts:  ``sample_hidden_states[bonus_idx]``.
        - Spec, drafts, all accepted:  bonus position.
        - Spec, drafts, partially accepted:  last accepted draft pos.
        """
        if spec_decode_metadata is None:
            # Non-spec-decode: simple 1-to-1 mapping
            if i < sample_hidden_states.shape[0]:
                return sample_hidden_states[i]
            return None

        num_drafts = spec_decode_metadata.num_draft_tokens[i]
        bonus_idx = spec_decode_metadata.bonus_logits_indices[i]
        if isinstance(bonus_idx, torch.Tensor):
            bonus_idx = bonus_idx.item()

        if num_drafts == 0:
            # No drafts → bonus position == normal decode position
            if 0 <= bonus_idx < sample_hidden_states.shape[0]:
                return sample_hidden_states[bonus_idx]
            return None

        # Has drafts → determine from accept length
        accept_len = len(valid_sampled_token_ids[i])
        if accept_len >= num_drafts + 1:
            # All accepted + bonus → use bonus position
            if 0 <= bonus_idx < sample_hidden_states.shape[0]:
                return sample_hidden_states[bonus_idx]
        elif accept_len > 0:
            # Partially accepted → last accepted draft position
            # Draft positions: [bonus_idx - num_drafts .. bonus_idx - 1]
            last_idx = bonus_idx - num_drafts + accept_len - 1
            if 0 <= last_idx < sample_hidden_states.shape[0]:
                return sample_hidden_states[last_idx]
        return None

    # main interface

    def generate_token_ids(
        self,
        valid_sampled_token_ids: List[List[int]],
        sampling_metadata: SamplingMetadata = None,
        scheduler_output: SchedulerOutput = None,
        spec_decode_metadata: SpecDecodeMetadata = None,
        positions: torch.Tensor = None,
        num_scheduled_tokens: int = 0,
        hidden_states: torch.Tensor = None,
        attn_metadata=None,
        aux_hidden_states: torch.Tensor = None,
    ) -> List[List[int]]:
        """Generate draft tokens via on-device hidden-state matching.

        Called by ``model_runner.propose_draft_token_ids()`` after the
        target model's forward pass.

        ``hidden_states`` should be **sample_hidden_states** (already
        indexed at logits positions) for correct per-request extraction.

        **Hot-loop invariant:** zero Ray / ZMQ / network calls in the
        steady state.  The only CPU work is a tiny scalar transfer
        ``(P, 2)`` and sub-µs numpy slices for draft tokens.
        """
        batch_size = len(valid_sampled_token_ids)
        if batch_size == 0:
            return []
        if hidden_states is None:
            return [[] for _ in range(batch_size)]

        input_batch = self.runner.input_batch

        # 1. Stable prompt_id + anchor hidden state per request
        prompt_ids: List[str] = []
        anchor_list: List[Optional[torch.Tensor]] = []

        for i in range(batch_size):
            req_id = input_batch.req_ids[i]
            req_state = self.runner.requests.get(req_id)
            if req_state is None:
                prompt_ids.append("")
                anchor_list.append(None)
                continue

            # Stable prompt_id from ORIGINAL prompt tokens (not token_ids_cpu)
            pid = prompt_id_from_token_ids(req_state.prompt_token_ids)
            prompt_ids.append(pid)

            # Anchor hidden state for this request
            hs = self._extract_anchor_hs(
                i, hidden_states, valid_sampled_token_ids,
                spec_decode_metadata,
            )
            anchor_list.append(hs)

        # 2. Consume ready prefetch futures (non-blocking).
        # prefetch_for_batch() was already called before the forward pass.
        # Here we poll for any newly-ready futures and fire for prompts
        # that might have arrived after the early prefetch.
        self._poll_pending()
        self._fire_prefetch_async(prompt_ids)

        # 3. On-device projection + similarity matching
        results: List[List[int]] = [[] for _ in range(batch_size)]
        # Accumulate on-device tensors to batch the single CPU sync
        pending: List[tuple] = []   # (batch_idx, sim_tensor, idx_tensor, cached)

        for i in range(batch_size):
            hs = anchor_list[i]
            pid = prompt_ids[i]
            if hs is None or pid not in self._cache:
                continue
            cached = self._cache[pid]
            if cached.n_entries == 0:
                continue
            if len(valid_sampled_token_ids[i]) < self.min_match_len:
                continue

            # All on NPU, no CPU sync
            # Cast bf16 → fp32 (Ascend NPU doesn't support bf16↔CPU)
            hs_f = hs.float()

            # Project: z = (h − μ) Wᵀ   →  (K,)
            z = (hs_f - cached.mean) @ cached.components.T

            # L2 normalise
            z = F.normalize(z, dim=0)

            # Cosine similarity with all stored keys
            sims = cached.keys @ z          # (n_entries,)
            best_sim, best_idx = sims.max(dim=0)

            pending.append((i, best_sim, best_idx, cached))

        if not pending:
            return results

        # 4. Single device → host sync for the whole batch
        sim_stack = torch.stack([p[1] for p in pending])    # (P,)
        idx_stack = torch.stack([p[2] for p in pending])    # (P,)
        sims_cpu = sim_stack.cpu().numpy()
        idxs_cpu = idx_stack.cpu().numpy()

        # 5. Draft token retrieval (CPU-only, O(1) per request)
        for j, (i, _, _, cached) in enumerate(pending):
            if sims_cpu[j] < self.similarity_threshold:
                continue

            # Adaptive window update
            req_id = input_batch.req_ids[i]
            accept_len = self._accept_lengths.get(req_id, 1)
            cached.update_window(accept_len)

            # Draft tokens from CPU cache (sub-µs numpy slice)
            draft = cached.get_draft_tokens(
                int(idxs_cpu[j]), cached.wnd_size,
            )
            if len(draft) > self.max_draft_tokens:
                draft = draft[: self.max_draft_tokens]
            results[i] = draft
            self._stat_hits += 1
            self._stat_total_draft_len += len(draft)

        self._stat_queries += len(pending)
        self._maybe_log_metrics()
        return results

    # bookkeeping

    def update_accept_lengths(
        self, req_ids: List[str], accept_lengths: List[int],
    ):
        """Update accept lengths for adaptive window control."""
        for rid, al in zip(req_ids, accept_lengths):
            self._accept_lengths[rid] = al
            self._stat_accept_sum += int(al)
            self._stat_accept_count += 1

    def clear_request(self, req_id: str):
        """Clear per-request state on completion."""
        self._accept_lengths.pop(req_id, None)

    # interface stubs

    def load_model(self, model):
        pass  # HSpec reuses the target model's hidden states

    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        skip_attn: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
    ):
        pass  # No separate model to warm up
