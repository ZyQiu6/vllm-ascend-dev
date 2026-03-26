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
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set

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
from vllm_ascend.spec_decode.hspec_utils import (
    hspec_profile_context_enabled,
    hspec_record_function,
    prompt_id_from_token_ids,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("HSPEC_LOG_LEVEL", os.getenv("VERL_LOGGING_LEVEL", "WARN")))

# Enable verbose HSpec debug logging when HSPEC_DEBUG is set.
HSPEC_DEBUG = os.getenv("HSPEC_DEBUG", "0") != "0"
# Per-step debug: only log one request per step to limit log volume.
HSPEC_DEBUG_REQ_IDX = int(os.getenv("HSPEC_DEBUG_REQ_IDX", "3"))

# Per-token generation breakdown timing (very verbose; use only for debugging).
# Enable with HSPEC_GEN=1. By default, trace the same request index as
# HSPEC_DEBUG_REQ_IDX; can override via HSPEC_GEN_REQ_IDX.
HSPEC_GEN = os.getenv("HSPEC_GEN", "0") != "0"
HSPEC_GEN_REQ_IDX = int(os.getenv("HSPEC_GEN_REQ_IDX",
                                  os.getenv("HSPEC_DEBUG_REQ_IDX", "3")))
HSPEC_GEN_MAX_CALLS = int(os.getenv("HSPEC_GEN_MAX_CALLS", "0"))  # 0 = no limit

# HistoSpec-comparison study: compare against an exact n-gram match baseline.
HSPEC_ADVAN_NGRAM = int(os.getenv("HSPEC_ADVAN_NGRAM", "3"))


def _now_ns() -> int:
    # perf_counter_ns is monotonic and high resolution; safe for timings.
    return time.perf_counter_ns()


def _ns_to_ms(ns: int) -> float:
    return float(ns) / 1_000_000.0


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
        "rollout_entry_starts", "rollout_entry_lens",
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
        rollout_entry_starts: np.ndarray,
        rollout_entry_lens: np.ndarray,
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
        self.rollout_entry_starts = rollout_entry_starts  # (R,) int32, CPU
        self.rollout_entry_lens = rollout_entry_lens      # (R,) int32, CPU
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


# Helper function for detokenization (with internal debug when HSPEC_DEBUG=1)
def _detokenize_safe(tokenizer, token_ids) -> str:
    """Safely detokenize token_ids to text, returns '<decode_error>' on failure.

    token_ids may be list, numpy array, or contain numpy.int64/torch.Tensor
    elements; we normalize to list of Python ints so tokenizer.decode() works.
    """
    # if HSPEC_DEBUG:
    #     logger.info(
    #         "HSPEC DEBUG _detokenize_safe: tokenizer=%s, token_ids type=%s, len=%s",
    #         type(tokenizer).__name__ if tokenizer is not None else "None",
    #         type(token_ids).__name__ if token_ids is not None else "None",
    #         len(token_ids) if token_ids is not None else 0,
    #     )
    try:
        if tokenizer is None:
            if HSPEC_DEBUG:
                logger.info("HSPEC DEBUG _detokenize_safe: early return <no_tokenizer>")
            return "<no_tokenizer>"
        if token_ids is None:
            return "<empty>"
        # Flatten and coerce to Python ints (handles list, ndarray, and
        # elements that are numpy.int64 or torch.Tensor scalars)
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        ids = []
        for x in token_ids:
            if hasattr(x, "item"):
                ids.append(int(x.item()))
            else:
                ids.append(int(x))
        if not ids:
            return "<empty>"
        text = tokenizer.decode(ids, skip_special_tokens=False)
        # if HSPEC_DEBUG:
        #     logger.info("HSPEC DEBUG _detokenize_safe: decode ok len(text)=%d", len(text))
        return text
    except Exception as e:
        import traceback
        n = len(token_ids) if token_ids is not None else 0
        if HSPEC_DEBUG:
            logger.info(
                "HSPEC DEBUG _detokenize_safe: EXCEPTION %s: %s\n%s",
                type(e).__name__, e, traceback.format_exc(),
            )
        return f"<decode_error: {n} tokens, {type(e).__name__}: {e}>"


# Cache for tokenizer loaded from name/path (model_config.tokenizer is str in vLLM)
_tokenizer_cache: Dict[str, Any] = {}


def _get_tokenizer_safe(runner) -> Optional[Any]:
    """Safely get tokenizer from runner/model/config.

    vLLM's model_config.tokenizer is often a string (model name/path), not an
    instance. When we get a str, we load via AutoTokenizer.from_pretrained(...)
    and cache by that path so decode works in debug.
    """
    global _tokenizer_cache
    if HSPEC_DEBUG:
        logger.info(
            "HSPEC DEBUG _get_tokenizer_safe: runner=%s, has model=%s, has vllm_config=%s, has tokenizer=%s",
            type(runner).__name__,
            hasattr(runner, "model"),
            hasattr(runner, "vllm_config"),
            hasattr(runner, "tokenizer"),
        )
    try:
        if hasattr(runner, "model") and runner.model is not None:
            if hasattr(runner.model, "tokenizer"):
                t = runner.model.tokenizer
                if not isinstance(t, str):
                    if HSPEC_DEBUG:
                        logger.info("HSPEC DEBUG _get_tokenizer_safe: got tokenizer from runner.model.tokenizer type=%s", type(t).__name__)
                    return t
        if hasattr(runner, "vllm_config") and runner.vllm_config is not None:
            mc = getattr(runner.vllm_config, "model_config", None)
            if mc is not None and hasattr(mc, "tokenizer"):
                t = mc.tokenizer
                if isinstance(t, str):
                    # model_config.tokenizer is the tokenizer name/path; load and cache
                    if t not in _tokenizer_cache:
                        try:
                            from transformers import AutoTokenizer
                            _tokenizer_cache[t] = AutoTokenizer.from_pretrained(t, trust_remote_code=True)
                        except Exception as load_err:
                            if HSPEC_DEBUG:
                                logger.warning("HSPEC DEBUG _get_tokenizer_safe: AutoTokenizer.from_pretrained(%r) failed: %s", t, load_err)
                            return None
                    t = _tokenizer_cache[t]
                if HSPEC_DEBUG:
                    logger.info("HSPEC DEBUG _get_tokenizer_safe: got tokenizer from vllm_config.model_config.tokenizer type=%s", type(t).__name__)
                return t    # right exit point
        if hasattr(runner, "tokenizer"):
            t = runner.tokenizer
            if not isinstance(t, str):
                if HSPEC_DEBUG:
                    logger.info("HSPEC DEBUG _get_tokenizer_safe: got tokenizer from runner.tokenizer type=%s", type(t).__name__)
                return t
        if HSPEC_DEBUG:
            logger.info("HSPEC DEBUG _get_tokenizer_safe: no tokenizer found, returning None")
        return None
    except Exception as e:
        import traceback
        if HSPEC_DEBUG:
            logger.info(
                "HSPEC DEBUG _get_tokenizer_safe: EXCEPTION %s: %s\n%s",
                type(e).__name__, e, traceback.format_exc(),
            )
        return None

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
        # req_id -> matched entry metadata for the *next* verification step.
        self._pending_verify_meta: Dict[str, Dict[str, int]] = {}
        # req_id -> stable prompt_id cache to avoid repeated hashing of the
        # same prompt token ids across decode steps for a live request.
        self._req_prompt_ids: Dict[str, str] = {}
        # Batch-aligned prompt-id cache. prefetch_for_batch() runs before every
        # forward pass, so generate_token_ids() can usually reuse this exact
        # alignment with just a req-id tuple comparison.
        self._cached_batch_req_ids: tuple[str, ...] = ()
        self._cached_batch_prompt_ids: List[str] = []
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

        # Low-frequency metrics reporting to table actors (so trainer-side
        # hspec_tables.compute_metrics() reflects worker-local online queries).
        self._report_every_calls = int(os.environ.get("HSPEC_REPORT_EVERY_CALLS", "200"))
        self._last_report_calls = 0
        self._reported_queries = 0
        self._reported_hits = 0
        self._reported_total_draft_len = 0

        # Entry-position study buffers. These are flushed asynchronously to the
        # global HSpec table actors at low frequency.
        self._entry_pending_match_count = 0
        self._entry_pending_delta_sum = 0
        self._entry_pending_abs_delta_sum = 0
        self._entry_pending_verify_count = 0
        self._entry_pending_accept_count = 0
        self._entry_pending_accept_len_sum = 0
        self._entry_pending_abs_delta_verify = defaultdict(int)
        self._entry_pending_abs_delta_accept = defaultdict(int)
        self._entry_pending_abs_delta_accept_len_sum = defaultdict(int)

        logger.info(
            "HSpec proposer initialised: threshold=%.3f, "
            "max_draft=%d, cache_cap=%d, fully_batched_match=1",
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
        prompt_ids = self._get_prompt_ids_for_batch(req_ids)
        prompt_ids = [pid for pid in prompt_ids if pid]
        if not prompt_ids:
            return

        # Consume any futures that became ready since last call
        self._poll_pending()
        # Fire new async fetches for cache misses
        self._fire_prefetch_async(prompt_ids)

    def _poll_pending(self) -> None:
        """Non-blocking: consume any ready prefetch futures.

        Uses ``ray.wait(timeout=0)`` which returns immediately with
        whatever futures are already completed.
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

    def _get_or_create_prompt_id(self, req_id: str) -> str:
        """Return a stable prompt_id for a live request, caching by req_id."""
        cached = self._req_prompt_ids.get(req_id)
        if cached is not None:
            return cached

        req_state = self.runner.requests.get(req_id)
        if req_state is None:
            return ""

        prompt_id = prompt_id_from_token_ids(req_state.prompt_token_ids)
        self._req_prompt_ids[req_id] = prompt_id
        return prompt_id

    def _get_prompt_ids_for_batch(self, req_ids: List[str]) -> List[str]:
        """Return prompt_ids aligned to the current scheduled batch."""
        batch_req_ids = tuple(str(req_id) for req_id in req_ids)
        if batch_req_ids == self._cached_batch_req_ids:
            return self._cached_batch_prompt_ids

        prompt_ids = [
            self._get_or_create_prompt_id(req_id)
            for req_id in batch_req_ids
        ]
        self._cached_batch_req_ids = batch_req_ids
        self._cached_batch_prompt_ids = prompt_ids
        return prompt_ids

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
        '''
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
        '''

        # Also report stats to the global table group at low frequency so the
        # trainer can observe match_rate/avg_draft_len even when queries are
        # executed locally on the worker.
        self._maybe_report_metrics()

    def _maybe_report_metrics(self) -> None:
        """Non-blocking metrics reporting (fire-and-forget Ray RPC)."""
        if self._stat_calls - self._last_report_calls < self._report_every_calls:
            return
        self._last_report_calls = self._stat_calls

        dq = int(self._stat_queries - self._reported_queries)
        dh = int(self._stat_hits - self._reported_hits)
        ddl = int(self._stat_total_draft_len - self._reported_total_draft_len)
        has_entry_pending = (
            self._entry_pending_match_count > 0
            or self._entry_pending_verify_count > 0
            or self._entry_pending_accept_count > 0
            or bool(self._entry_pending_abs_delta_verify)
            or bool(self._entry_pending_abs_delta_accept)
            or bool(self._entry_pending_abs_delta_accept_len_sum)
        )
        if dq <= 0 and dh <= 0 and ddl <= 0 and not has_entry_pending:
            return

        self._reported_queries = int(self._stat_queries)
        self._reported_hits = int(self._stat_hits)
        self._reported_total_draft_len = int(self._stat_total_draft_len)

        try:
            # Fire-and-forget, never block the hot loop.
            if hasattr(self.hspec_tables, "report_online_metrics_async"):
                self.hspec_tables.report_online_metrics_async(dq, dh, ddl)
        except Exception:
            # Swallow all errors; metrics must never affect decoding.
            pass

        if has_entry_pending:
            try:
                if hasattr(self.hspec_tables, "report_entry_metrics_async"):
                    self.hspec_tables.report_entry_metrics_async(
                        match_count=int(self._entry_pending_match_count),
                        delta_sum=int(self._entry_pending_delta_sum),
                        abs_delta_sum=int(self._entry_pending_abs_delta_sum),
                        verify_count=int(self._entry_pending_verify_count),
                        accept_count=int(self._entry_pending_accept_count),
                        accept_len_sum=int(self._entry_pending_accept_len_sum),
                        abs_delta_verify=dict(self._entry_pending_abs_delta_verify),
                        abs_delta_accept=dict(self._entry_pending_abs_delta_accept),
                        abs_delta_accept_len_sum=dict(
                            self._entry_pending_abs_delta_accept_len_sum
                        ),
                    )
            except Exception:
                pass
            finally:
                self._entry_pending_match_count = 0
                self._entry_pending_delta_sum = 0
                self._entry_pending_abs_delta_sum = 0
                self._entry_pending_verify_count = 0
                self._entry_pending_accept_count = 0
                self._entry_pending_accept_len_sum = 0
                self._entry_pending_abs_delta_verify.clear()
                self._entry_pending_abs_delta_accept.clear()
                self._entry_pending_abs_delta_accept_len_sum.clear()

    @staticmethod
    def _build_rollout_entry_spans(
        entry_rollout_idx: np.ndarray,
        n_entries: int,
        num_rollouts: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build contiguous per-rollout entry spans for local-window matching."""
        starts = np.zeros((num_rollouts,), dtype=np.int32)
        lens = np.zeros((num_rollouts,), dtype=np.int32)
        if n_entries <= 0 or num_rollouts <= 0:
            return starts, lens

        counts = np.bincount(
            entry_rollout_idx[:n_entries],
            minlength=num_rollouts,
        ).astype(np.int32, copy=False)
        cursor = 0
        for ridx, count in enumerate(counts.tolist()):
            starts[ridx] = cursor
            lens[ridx] = count
            cursor += count
        return starts, lens

    def _build_cached_table(self, data: dict) -> _CachedPromptTable:
        """Convert serialised table data dict → on-device cached table."""
        # NOTE: Ray may deserialize numpy arrays as non-writable (read-only)
        # views; torch.from_numpy warns about undefined behaviour on write.
        # These tensors are read-only in our usage, but we still copy here to
        # silence warnings and keep behaviour well-defined. This is *prefetch*
        # (not in the hot loop).
        mean_np = np.array(data["mean"], dtype=np.float32, copy=True)
        comp_np = np.array(data["components"], dtype=np.float32, copy=True)
        keys_np = np.array(data["keys"], dtype=np.float32, copy=True)

        mean = torch.from_numpy(mean_np).to(self.device, non_blocking=True)

        components = torch.from_numpy(comp_np).to(self.device, non_blocking=True)

        keys = torch.from_numpy(keys_np).to(self.device, non_blocking=True)

        # Ensure rollout_seqs are numpy arrays on CPU
        rollout_seqs = []
        for s in data["rollout_seqs"]:
            if isinstance(s, np.ndarray):
                rollout_seqs.append(s)
            else:
                rollout_seqs.append(np.asarray(s, dtype=np.int32))

        entry_rollout_idx = np.asarray(
            data["entry_rollout_idx"], dtype=np.int32,
        )
        n_entries = int(data["n_entries"])
        rollout_entry_starts, rollout_entry_lens = self._build_rollout_entry_spans(
            entry_rollout_idx,
            n_entries,
            len(rollout_seqs),
        )

        return _CachedPromptTable(
            mean=mean,
            components=components,
            keys=keys,
            rollout_seqs=rollout_seqs,
            entry_rollout_idx=entry_rollout_idx,
            entry_offset=np.asarray(data["entry_offset"], dtype=np.int32),
            rollout_entry_starts=rollout_entry_starts,
            rollout_entry_lens=rollout_entry_lens,
            n_entries=n_entries,
            wnd_size=int(data.get("wnd_size", 8)),
            max_wnd=int(data.get("max_wnd", 28)),
            min_wnd=int(data.get("min_wnd", 2)),
        )

    @staticmethod
    def _window_base_pos(decoded_len: int) -> int:
        # After accepting `decoded_len` response tokens, the next query uses the
        # anchor at local key position `decoded_len - 1`.
        return max(int(decoded_len) - 1, 0)

    def _build_batched_table_tensors(
        self,
        cached_tables: List[_CachedPromptTable],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack ragged per-prompt components/keys into padded batch tensors."""
        num_rows = len(cached_tables)
        if num_rows == 0:
            empty = torch.empty((0,), dtype=dtype, device=device)
            return empty, empty, torch.empty((0,), dtype=torch.long, device=device)

        k_max = max(int(cached.components.shape[0]) for cached in cached_tables)
        m_max = max(int(cached.n_entries) for cached in cached_tables)
        hidden_dim = int(cached_tables[0].mean.shape[0])

        comp_batch = torch.zeros(
            (num_rows, k_max, hidden_dim),
            dtype=dtype,
            device=device,
        )
        keys_batch = torch.zeros(
            (num_rows, m_max, k_max),
            dtype=dtype,
            device=device,
        )
        key_lengths = torch.empty(
            (num_rows,),
            dtype=torch.long,
            device=device,
        )

        for row, cached in enumerate(cached_tables):
            k_i = int(cached.components.shape[0])
            m_i = int(cached.n_entries)
            comp_batch[row, :k_i, :] = cached.components.to(dtype=dtype)
            if m_i > 0:
                keys_batch[row, :m_i, :k_i] = cached.keys[:m_i, :k_i].to(dtype=dtype)
            key_lengths[row] = m_i

        return comp_batch, keys_batch, key_lengths

    def _match_projected_batch(
        self,
        batch_indices: List[int],
        cached_tables: List[_CachedPromptTable],
        z_batch: torch.Tensor,
        keys_batch: torch.Tensor,
        key_lengths: torch.Tensor,
        decoded_lens: List[int],
    ) -> List[tuple[int, torch.Tensor, torch.Tensor, _CachedPromptTable, int]]:
        """Fully batched similarity matching with padding + mask over ragged M."""
        if not batch_indices:
            return []
        sims = torch.bmm(keys_batch, z_batch.unsqueeze(-1)).squeeze(-1)
        if keys_batch.shape[1] > 0:
            positions = torch.arange(
                keys_batch.shape[1], device=z_batch.device,
            ).unsqueeze(0)
            invalid_mask = positions >= key_lengths.unsqueeze(1)
            sims = sims.masked_fill(invalid_mask, torch.finfo(sims.dtype).min)
        best_sims, best_idxs = sims.max(dim=1)
        return [
            (
                batch_idx,
                best_sims[row],
                best_idxs[row],
                cached,
                self._window_base_pos(decoded_lens[batch_idx]),
            )
            for row, (batch_idx, cached) in enumerate(zip(batch_indices, cached_tables))
        ]

    @staticmethod
    def _has_same_histo_ngram(
        current_tokens: List[int],
        matched_seq: np.ndarray,
        matched_pos: int,
    ) -> bool:
        """Whether HistoSpec could have matched using the same exact n-gram."""
        n = int(HSPEC_ADVAN_NGRAM)
        if n <= 0:
            return False
        if len(current_tokens) < n:
            return False

        hist_end = int(matched_pos) + 1
        if hist_end < n:
            return False

        current_ngram = [int(x) for x in current_tokens[-n:]]
        hist_ngram = matched_seq[hist_end - n:hist_end].tolist()
        return current_ngram == [int(x) for x in hist_ngram]

    # anchor hidden-state extraction

    @staticmethod
    def _extract_anchor_hs_batch(
        sample_hidden_states: torch.Tensor,
        valid_sampled_token_ids: List[List[int]],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
    ) -> tuple[torch.Tensor, List[int]]:
        """Batch-extract all valid anchor hidden states in one gather."""
        batch_size = len(valid_sampled_token_ids)
        if batch_size == 0:
            return sample_hidden_states[:0], []

        device = sample_hidden_states.device
        accept_lens = torch.tensor(
            [len(sampled_ids) for sampled_ids in valid_sampled_token_ids],
            dtype=torch.long,
            device=device,
        )
        anchor_indices = torch.full_like(accept_lens, -1)

        if spec_decode_metadata is None:
            candidate_indices = torch.arange(
                batch_size, dtype=torch.long, device=device,
            )
            valid_mask = (
                (accept_lens > 0)
                & (candidate_indices < sample_hidden_states.shape[0])
            )
            anchor_indices[valid_mask] = candidate_indices[valid_mask]
        else:
            num_drafts = torch.as_tensor(
                spec_decode_metadata.num_draft_tokens[:batch_size],
                dtype=torch.long,
                device=device,
            )
            bonus_indices = spec_decode_metadata.bonus_logits_indices[
                :batch_size
            ].to(device=device, dtype=torch.long)

            has_accept = accept_lens > 0
            no_draft = num_drafts == 0
            has_draft = num_drafts > 0
            full_accept = has_draft & (accept_lens >= (num_drafts + 1))
            partial_accept = has_draft & has_accept & (~full_accept)

            anchor_indices[has_accept & no_draft] = bonus_indices[
                has_accept & no_draft
            ]
            anchor_indices[full_accept] = bonus_indices[full_accept]
            anchor_indices[partial_accept] = (
                bonus_indices[partial_accept]
                - num_drafts[partial_accept]
                + accept_lens[partial_accept]
                - 1
            )

            valid_mask = (
                (anchor_indices >= 0)
                & (anchor_indices < sample_hidden_states.shape[0])
            )

        if not torch.any(valid_mask):
            return sample_hidden_states[:0], []

        gathered = sample_hidden_states.index_select(0, anchor_indices[valid_mask])
        valid_batch_indices = torch.nonzero(
            valid_mask, as_tuple=False,
        ).flatten().tolist()
        return gathered, valid_batch_indices

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
        # NOTE: HSPEC_GEN is intentionally implemented to be *zero-cost* when
        # disabled (only a single boolean check in hot loop).
        batch_size = len(valid_sampled_token_ids)
        if batch_size == 0:
            return []
        if hidden_states is None:
            return [[] for _ in range(batch_size)]
        input_batch = self.runner.input_batch

        gen_enabled = HSPEC_GEN
        gen_req_idx = HSPEC_GEN_REQ_IDX
        if gen_enabled and HSPEC_GEN_MAX_CALLS > 0:
            # Hard cap to avoid runaway logs in long trainings.
            # We use _stat_calls as a monotonic per-call counter already
            # maintained by this proposer.
            if getattr(self, "_stat_calls", 0) >= HSPEC_GEN_MAX_CALLS:
                gen_enabled = False
        prof_enabled = hspec_profile_context_enabled()

        # 1. Stable prompt_id + batch anchor hidden states
        req_ids = list(input_batch.req_ids[:batch_size])
        req_states = [self.runner.requests.get(req_id) for req_id in req_ids]

        t0_pid = _now_ns() if gen_enabled else 0
        with (hspec_record_function("hspec/proposal/prompt_id")
              if prof_enabled else nullcontext()):
            prompt_ids = self._get_prompt_ids_for_batch(req_ids)
        t1_pid = _now_ns() if gen_enabled else 0

        decoded_lens = [
            len(getattr(req_state, "output_token_ids", []))
            if req_state is not None else 0
            for req_state in req_states
        ]

        t0_extract = _now_ns() if gen_enabled else 0
        with (hspec_record_function("hspec/proposal/extract_anchor_hs")
              if prof_enabled else nullcontext()):
            anchor_hidden_states, anchor_batch_indices = (
                self._extract_anchor_hs_batch(
                    hidden_states,
                    valid_sampled_token_ids,
                    spec_decode_metadata,
                )
            )
        t1_extract = _now_ns() if gen_enabled else 0

        anchor_list: List[Optional[torch.Tensor]] = [None] * batch_size
        for slot, batch_idx in enumerate(anchor_batch_indices):
            anchor_list[batch_idx] = anchor_hidden_states[slot]

        if gen_enabled:
            self._hspec_gen_timing = {
                "prompt_id_ms": _ns_to_ms(t1_pid - t0_pid) if t0_pid else 0.0,
                "extract_anchor_hs_ms": _ns_to_ms(t1_extract - t0_extract) if t0_extract else 0.0,
                "anchor_total_ms": _ns_to_ms(t1_extract - t0_pid) if t0_pid else 0.0,
            }

        if HSPEC_DEBUG:
            # (1)(2) One prompt per step: only the chosen request
            # Log correlation fields (req_id, prompt_id, decoded_len, decoded_tokens)
            try:
                di = min(HSPEC_DEBUG_REQ_IDX, batch_size - 1) if batch_size else 0
                req_id = req_ids[di]
                req_state = req_states[di]
                if req_state is None:
                    decoded_tokens = []
                    prompt_tokens = []
                else:
                    decoded_tokens = list(getattr(req_state, "output_token_ids", []))
                    prompt_tokens = list(getattr(req_state, "prompt_token_ids", []))
                decoded_len = len(decoded_tokens)
                # Correlation id: same (req_id, prompt_id, decoded_len) in STEP_BEGIN and CACHE_MATCH
                corr_id = f"req_id={req_id!r} prompt_id={prompt_ids[di]!r} decoded_len={decoded_len}"
                anchor_hs_summary = "anchor_hs=None"
                if di < len(anchor_list) and anchor_list[di] is not None:
                    hs = anchor_list[di]
                    try:
                        norm = float(hs.float().norm().item())
                    except Exception:
                        norm = float("nan")
                    anchor_hs_summary = (
                        f"anchor_hs.shape={tuple(hs.shape)} norm={norm:.6f} device={hs.device}"
                    )
                # Detokenize prompt_tokens + decoded_tokens
                tokenizer = _get_tokenizer_safe(self.runner)
                prompt_decoded_text = _detokenize_safe(tokenizer, prompt_tokens + decoded_tokens)
                logger.info(
                    "------------------------------------------- generate_token_ids() STEP_BEGIN -------------------------------------------\n"
                    "HSPEC DEBUG STEP_BEGIN [req_idx=%d] %s | "
                    "decoded_tokens=%s | accepted_step_tokens=%s | "
                    "hidden_states.shape=%s dtype=%s device=%s | prompt_tokens=%s | "
                    "prompt+decoded_text=%r | %s",
                    di,
                    corr_id,
                    decoded_tokens,
                    valid_sampled_token_ids[di] if di < len(valid_sampled_token_ids) else [],
                    tuple(hidden_states.shape),
                    str(hidden_states.dtype),
                    hidden_states.device,
                    prompt_tokens,
                    prompt_decoded_text,
                    anchor_hs_summary,
                )
            except Exception:
                logger.exception("HSPEC DEBUG: failed to log prompt/hidden_state info")

        # 2. Consume ready prefetch futures (non-blocking).
        # prefetch_for_batch() was already called before the forward pass.
        # Here we poll for any newly-ready futures and fire for prompts
        # that might have arrived after the early prefetch.
        t0_poll = _now_ns() if gen_enabled else 0
        with hspec_record_function("hspec/proposal/poll_pending"):
            self._poll_pending()
        t1_poll = _now_ns() if gen_enabled else 0
        t0_fire = _now_ns() if gen_enabled else 0
        with hspec_record_function("hspec/proposal/fire_prefetch_async"):
            self._fire_prefetch_async(prompt_ids)
        t1_fire = _now_ns() if gen_enabled else 0

        # 3. On-device projection + similarity matching
        results: List[List[int]] = [[] for _ in range(batch_size)]
        # Accumulate on-device tensors to batch the single CPU sync
        pending: List[tuple] = []   # (batch_idx, sim_tensor, idx_tensor, cached, base_pos)
        trace_pending_j: Optional[int] = None
        trace_skip_reason: Optional[str] = None
        active_match_indices: List[int] = []
        active_cached_tables: List[_CachedPromptTable] = []
        trace_candidate_selected = False

        for i in range(batch_size):
            hs = anchor_list[i]
            pid = prompt_ids[i]
            if hs is None:
                if gen_enabled and i == gen_req_idx:
                    trace_skip_reason = "anchor_none"
                continue
            if pid not in self._cache:
                if gen_enabled and i == gen_req_idx:
                    trace_skip_reason = "prompt_not_cached"
                continue
            cached = self._cache[pid]
            if cached.n_entries == 0:
                if gen_enabled and i == gen_req_idx:
                    trace_skip_reason = "empty_prompt_table"
                continue
            if len(valid_sampled_token_ids[i]) < self.min_match_len:
                if gen_enabled and i == gen_req_idx:
                    trace_skip_reason = (
                        f"below_min_match_len:{len(valid_sampled_token_ids[i])}"
                    )
                continue

            active_match_indices.append(i)
            active_cached_tables.append(cached)
            if gen_enabled and i == gen_req_idx:
                trace_candidate_selected = True

        if active_match_indices:
            t0_cast = _now_ns() if gen_enabled else 0
            with (hspec_record_function("hspec/proposal/project_and_match", use_npu_stream=True)
                  if prof_enabled else nullcontext()):
                active_anchor_hs = torch.stack(
                    [anchor_list[i] for i in active_match_indices]
                ).float()
                t1_cast = _now_ns() if gen_enabled else 0

                t0_proj = _now_ns() if gen_enabled else 0
                mean_batch = torch.stack(
                    [cached.mean for cached in active_cached_tables]
                )
                comp_batch, keys_batch, key_lengths = self._build_batched_table_tensors(
                    active_cached_tables,
                    dtype=active_anchor_hs.dtype,
                    device=active_anchor_hs.device,
                )
                z_batch = torch.bmm(
                    (active_anchor_hs - mean_batch).unsqueeze(1),
                    comp_batch.transpose(1, 2),
                ).squeeze(1)
                t1_proj = _now_ns() if gen_enabled else 0

                t0_sim = _now_ns() if gen_enabled else 0
                pending.extend(
                    self._match_projected_batch(
                        active_match_indices,
                        active_cached_tables,
                        z_batch,
                        keys_batch,
                        key_lengths,
                        decoded_lens,
                    )
                )
                t1_sim = _now_ns() if gen_enabled else 0

            if gen_enabled:
                for j, (batch_idx, _, _, _, _) in enumerate(pending):
                    if batch_idx == gen_req_idx:
                        trace_pending_j = j
                        break
                if trace_candidate_selected and trace_pending_j is None:
                    trace_skip_reason = "empty_entry_window"
                td = getattr(self, "_hspec_gen_timing", {}) if hasattr(self, "_hspec_gen_timing") else {}
                td.update({
                    "cast_fp32_ms": _ns_to_ms(t1_cast - t0_cast) if t0_cast else 0.0,
                    "pca_project_ms": _ns_to_ms(t1_proj - t0_proj) if t0_proj else 0.0,
                    "similarity_ms": _ns_to_ms(t1_sim - t0_sim) if t0_sim else 0.0,
                })
                self._hspec_gen_timing = td

        if HSPEC_DEBUG:
            try:
                di = min(HSPEC_DEBUG_REQ_IDX, batch_size - 1) if batch_size else 0
                req_id = req_ids[di]
                req_state = req_states[di]
                decoded_tokens_at_match = []
                if req_state is not None:
                    decoded_tokens_at_match = list(getattr(req_state, "output_token_ids", []))
                decoded_len_at_match = len(decoded_tokens_at_match)
                corr_id = f"req_id={req_id!r} prompt_id={prompt_ids[di]!r} decoded_len={decoded_len_at_match}"

                # Pending list: only the chosen request if it is in pending
                pending_line = None
                cached_for_debug = None
                best_idx_val = -1
                sim_val = float("nan")
                for (i, sim_t, idx_t, cached, _) in pending:
                    if i != di:
                        continue
                    cached_for_debug = cached
                    try:
                        sim_val = float(sim_t.detach().float().item())
                    except Exception:
                        sim_val = float("nan")
                    try:
                        best_idx_val = int(idx_t.detach().item())
                    except Exception:
                        best_idx_val = -1
                    pending_line = (
                        f"CACHE_MATCH [req_idx={di}] {corr_id} | "
                        f"decoded_tokens_at_match={decoded_tokens_at_match} | "
                        f"best_idx={best_idx_val} sim={sim_val:.6f} n_entries={cached.n_entries}"
                    )
                    break
                if pending_line is None:
                    pending_line = (
                        f"CACHE_MATCH [req_idx={di}] {corr_id} | "
                        f"decoded_tokens_at_match={decoded_tokens_at_match} | "
                        f"not_in_pending (pending_size={len(pending)})"
                    )
                else:
                    # (1) Print cache table contents (all entries) with correlation to decoded state
                    if cached_for_debug is not None:
                        table_info_lines = []
                        n_entries = cached_for_debug.n_entries
                        table_info_lines.append(
                            f"cache_table {corr_id} | decoded_tokens_at_match={decoded_tokens_at_match} | "
                            f"n_entries={n_entries} wnd_size={cached_for_debug.wnd_size} "
                            f"mean.shape={tuple(cached_for_debug.mean.shape)} "
                            f"components.shape={tuple(cached_for_debug.components.shape)} "
                            f"keys.shape={tuple(cached_for_debug.keys.shape)} | "
                            f"best_hit_entry={best_idx_val} → draft_tokens_for_next_step (see entry below)"
                        )
                        tokenizer = _get_tokenizer_safe(self.runner)
                        for entry_idx in range(n_entries):
                            ridx = int(cached_for_debug.entry_rollout_idx[entry_idx])
                            off = int(cached_for_debug.entry_offset[entry_idx])
                            seq = cached_for_debug.rollout_seqs[ridx]
                            draft_tokens = seq[off: off + cached_for_debug.wnd_size].tolist()
                            # Detokenize rollout sequence
                            rollout_text = _detokenize_safe(tokenizer, draft_tokens)
                            # Get similarity for this entry (if keys available)
                            try:
                                # Use the projected anchor_hs to compute similarity
                                if di < len(anchor_list) and anchor_list[di] is not None:
                                    hs_f = anchor_list[di].float()
                                    z = (hs_f - cached_for_debug.mean) @ cached_for_debug.components.T
                                    # z = F.normalize(z, dim=0)
                                    entry_sim = float((cached_for_debug.keys[entry_idx] @ z).item())
                                else:
                                    entry_sim = float("nan")
                            except Exception:
                                entry_sim = float("nan")
                            table_info_lines.append(
                                f"  entry[{entry_idx}] rollout_idx={ridx} offset={off} "
                                f"sim={entry_sim:.4f} | "
                                f"decoded_tokens_at_match={decoded_tokens_at_match} → draft_tokens={draft_tokens} "
                                f"rollout_text={rollout_text!r}"
                            )
                        pending_line += "\n" + "\n".join(table_info_lines)
                logger.info(
                    "------------------------------------------- generate_token_ids() CACHE_MATCH -------------------------------------------\n"
                    "HSPEC DEBUG %s",
                    pending_line,
                )
            except Exception:
                logger.exception("HSPEC DEBUG: failed to log pending list")

        if not pending:
            if gen_enabled and batch_size > 0:
                di = min(gen_req_idx, batch_size - 1)
                try:
                    req_id = req_ids[di]
                    req_state = req_states[di]
                    decoded_len = len(getattr(req_state, "output_token_ids", [])) if req_state is not None else -1
                    pid = prompt_ids[di] if di < len(prompt_ids) else ""
                    td = getattr(self, "_hspec_gen_timing", {}) if hasattr(self, "_hspec_gen_timing") else {}
                    td.update({
                        "poll_pending_ms": _ns_to_ms(t1_poll - t0_poll) if t0_poll else 0.0,
                        "fire_prefetch_async_ms": _ns_to_ms(t1_fire - t0_fire) if t0_fire else 0.0,
                    })
                    self._hspec_gen_timing = td
                    logger.warning(
                        "HSPEC GEN proposer_breakdown [req_idx=%d] req_id=%s prompt_id=%s decoded_len=%d "
                        "status=no_pending skip_reason=%s timing_ms=%s",
                        int(di),
                        str(req_id),
                        str(pid),
                        int(decoded_len),
                        str(trace_skip_reason),
                        td,
                    )
                except Exception:
                    pass
            if HSPEC_DEBUG:
                logger.info("HSPEC DEBUG generate_token_ids(): no pending entries, all requests miss cache/anchor")
            return results

        # 4. Single device → host sync for the whole batch
        t0_stack = _now_ns() if gen_enabled else 0
        with hspec_record_function("hspec/proposal/device_to_host_sync", use_npu_stream=True):
            sim_stack = torch.stack([p[1] for p in pending])    # (P,)
            idx_stack = torch.stack([p[2] for p in pending])    # (P,)
            t1_stack = _now_ns() if gen_enabled else 0
            t0_copy = _now_ns() if gen_enabled else 0
            sims_cpu = sim_stack.cpu().numpy()
            idxs_cpu = idx_stack.cpu().numpy()
            t1_copy = _now_ns() if gen_enabled else 0

        # 5. Draft token retrieval (CPU-only, O(1) per request)
        for j, (i, _, _, cached, base_pos) in enumerate(pending):
            t0_retrieve = _now_ns() if (gen_enabled and i == gen_req_idx) else 0
            if sims_cpu[j] < self.similarity_threshold:
                continue

            # Adaptive window update
            req_id = req_ids[i]
            accept_len = self._accept_lengths.get(req_id, 1)
            cached.update_window(accept_len)

            # Draft tokens from CPU cache (sub-µs numpy slice)
            prof_this_req = prof_enabled
            with (hspec_record_function("hspec/proposal/draft_retrieve")
                  if prof_this_req else nullcontext()):
                draft = cached.get_draft_tokens(
                    int(idxs_cpu[j]), cached.wnd_size,
                )
            if len(draft) > self.max_draft_tokens:
                draft = draft[: self.max_draft_tokens]
            results[i] = draft
            self._stat_hits += 1
            self._stat_total_draft_len += len(draft)

            if draft:
                req_id = req_ids[i]
                matched_entry_idx = int(idxs_cpu[j])
                matched_rollout_idx = int(cached.entry_rollout_idx[matched_entry_idx])
                matched_pos = int(cached.entry_offset[matched_entry_idx]) - 1
                delta = int(matched_pos - int(base_pos))
                abs_delta = abs(delta)
                req_state = req_states[i]
                current_tokens = (
                    list(getattr(req_state, "output_token_ids", []))
                    if req_state is not None else []
                )
                histo_ngram_match = self._has_same_histo_ngram(
                    current_tokens,
                    cached.rollout_seqs[matched_rollout_idx],
                    matched_pos,
                )
                self._entry_pending_match_count += 1
                self._entry_pending_delta_sum += delta
                self._entry_pending_abs_delta_sum += abs_delta
                self._pending_verify_meta[req_id] = {
                    "delta": delta,
                    "abs_delta": abs_delta,
                    "base_pos": int(base_pos),
                    "matched_pos": matched_pos,
                    "matched_rollout_idx": matched_rollout_idx,
                    "histo_ngram_match": int(histo_ngram_match),
                }

            t1_retrieve = _now_ns() if (gen_enabled and i == gen_req_idx) else 0
            if gen_enabled and i == gen_req_idx:
                td = getattr(self, "_hspec_gen_timing", {}) if hasattr(self, "_hspec_gen_timing") else {}
                td.update({
                    "poll_pending_ms": _ns_to_ms(t1_poll - t0_poll) if t0_poll else 0.0,
                    "fire_prefetch_async_ms": _ns_to_ms(t1_fire - t0_fire) if t0_fire else 0.0,
                    "stack_scalars_ms": _ns_to_ms(t1_stack - t0_stack) if t0_stack else 0.0,
                    "device_to_host_ms": _ns_to_ms(t1_copy - t0_copy) if t0_copy else 0.0,
                    "draft_retrieve_ms": _ns_to_ms(t1_retrieve - t0_retrieve) if t0_retrieve else 0.0,
                })
                self._hspec_gen_timing = td

        self._stat_queries += len(pending)
        self._maybe_log_metrics()

        # HSPEC_GEN: per-token breakdown log for one traced request only.
        # We intentionally log at WARNING to make sure it lands in the main
        # training output when log levels filter INFO.
        if gen_enabled and batch_size > 0:
            di = min(gen_req_idx, batch_size - 1)
            try:
                req_id = req_ids[di]
                req_state = req_states[di]
                decoded_len = len(getattr(req_state, "output_token_ids", [])) if req_state is not None else -1
                pid = prompt_ids[di] if di < len(prompt_ids) else ""
                anchor = anchor_list[di] if di < len(anchor_list) else None
                anchor_norm = None
                if anchor is not None:
                    try:
                        anchor_norm = float(anchor.float().norm().item())
                    except Exception:
                        anchor_norm = None

                # Similarity info (if this request is in pending)
                sim_val = None
                best_idx_val = None
                n_entries = None
                wnd_size = None
                if trace_pending_j is not None:
                    # Find the pending slot for this batch index
                    if trace_pending_j < len(pending) and pending[trace_pending_j][0] == di:
                        sim_val = float(sims_cpu[trace_pending_j])
                        best_idx_val = int(idxs_cpu[trace_pending_j])
                        n_entries = int(pending[trace_pending_j][3].n_entries)
                        wnd_size = int(pending[trace_pending_j][3].wnd_size)

                draft = results[di] if di < len(results) else []
                td = getattr(self, "_hspec_gen_timing", {}) if hasattr(self, "_hspec_gen_timing") else {}
                logger.warning(
                    "HSPEC GEN proposer_breakdown [req_idx=%d] req_id=%s prompt_id=%s decoded_len=%d "
                    "anchor_norm=%s sim=%s best_idx=%s n_entries=%s wnd_size=%s draft_len=%d draft=%s "
                    "timing_ms=%s",
                    int(di),
                    str(req_id),
                    str(pid),
                    int(decoded_len),
                    "None" if anchor_norm is None else f"{anchor_norm:.6f}",
                    "None" if sim_val is None else f"{sim_val:.6f}",
                    "None" if best_idx_val is None else str(best_idx_val),
                    "None" if n_entries is None else str(n_entries),
                    "None" if wnd_size is None else str(wnd_size),
                    int(len(draft)),
                    list(draft),
                    td,
                )
            except Exception:
                # Never break decoding due to debug logging.
                pass

        if HSPEC_DEBUG:
            try:
                di = min(HSPEC_DEBUG_REQ_IDX, batch_size - 1) if batch_size else 0
                # Matched + draft for chosen request only
                matched_line = None
                for j, (i, _, _, cached, _) in enumerate(pending):
                    if i != di or sims_cpu[j] < self.similarity_threshold:
                        continue
                    matched_line = (
                        f"matched [req_idx={di}] prompt_id={prompt_ids[di]!r} "
                        f"sim={float(sims_cpu[j]):.4f} best_idx={int(idxs_cpu[j])} "
                        f"wnd_size={cached.wnd_size} draft_tokens={list(results[di])}"
                    )
                    break
                if matched_line is None:
                    matched_line = (
                        f"matched: req_idx={di} not above threshold (threshold=%.4f) or not in pending"
                        % self.similarity_threshold
                    )
                logger.info(
                    "HSPEC DEBUG generate_token_ids() [req_idx=%d]: %s | final draft_token_ids=%s",
                    di, matched_line, results[di] if di < len(results) else [],
                )
                logger.info(
                    "------------------------------------------- generate_token_ids() end -------------------------------------------\n"
                )
            except Exception:
                logger.exception("HSPEC DEBUG: failed to log draft / match info")

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

    def update_verification_outcomes(
        self,
        req_ids: List[str],
        accepted_prefix_lengths: List[int],
    ) -> tuple[int, int]:
        """Consume true draft-prefix acceptance outcomes for HSpec studies.

        ``accepted_prefix_lengths`` must be the verification-time
        prefix-match lengths between ``draft`` and ``out`` for each request.
        """
        accept_advan_count = 0
        reject_advan_count = 0
        for rid, accepted_prefix_len in zip(req_ids, accepted_prefix_lengths):
            meta = self._pending_verify_meta.pop(rid, None)
            if meta is None:
                continue
            abs_delta = int(meta["abs_delta"])
            apl = int(accepted_prefix_len)
            self._entry_pending_verify_count += 1
            self._entry_pending_accept_len_sum += apl
            self._entry_pending_abs_delta_verify[abs_delta] += 1
            if apl >= 1:
                self._entry_pending_accept_count += 1
                self._entry_pending_abs_delta_accept[abs_delta] += 1
                self._entry_pending_abs_delta_accept_len_sum[abs_delta] += apl
                if not bool(meta.get("histo_ngram_match", 0)):
                    accept_advan_count += 1
            else:
                if not bool(meta.get("histo_ngram_match", 0)):
                    reject_advan_count += 1
        return (accept_advan_count, reject_advan_count)

    def update_entry_verification_outcomes(
        self,
        req_ids: List[str],
        accepted_prefix_lengths: List[int],
    ) -> None:
        """Backward-compatible wrapper for entry-only callers."""
        self.update_verification_outcomes(req_ids, accepted_prefix_lengths)

    def clear_request(self, req_id: str):
        """Clear per-request state on completion."""
        self._accept_lengths.pop(req_id, None)
        self._pending_verify_meta.pop(req_id, None)
        self._req_prompt_ids.pop(req_id, None)
        if req_id in self._cached_batch_req_ids:
            self._cached_batch_req_ids = ()
            self._cached_batch_prompt_ids = []

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
