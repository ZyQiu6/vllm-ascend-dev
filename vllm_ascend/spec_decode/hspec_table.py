# Copyright 2025 HSpec Authors
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
HSpec: Hidden State based Speculative Decoding – Query Table.

High-performance table implementation with:
  - Continuous ndarray keys  (M, K)  float16, L2-normalised PCA projections
  - Reference-based value storage: O(L) per rollout instead of O(L²)
  - Per-prompt PCA parameters (μ, W) for on-device projection
  - Batch build / query interfaces for high throughput
  - Partitioned Ray actors with ZMQ RPC for distributed serving

Design-doc references:
  §4  table structure          §3.1 service layer
  §6  async build              §7   perf rules & metrics
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import msgpack
import numpy as np
import ray
import zmq

from vllm_ascend.spec_decode.hspec_utils import (
    PromptPCAParams,
    fit_pca_multi_sequence,
    fit_pca_single_sequence,
    stable_partition_id,
)

logger = logging.getLogger(__name__)


# Per-prompt table data  (continuous-array, reference-value storage)


class PromptTableData:

    __slots__ = (
        "pca_params",
        "keys",
        "rollout_seqs",
        "entry_rollout_idx",
        "entry_offset",
        "rewards",
        "n_entries",
        "max_entries",
        # adaptive window  (design-doc §5.3)
        "wnd_size",
        "max_wnd",
        "min_wnd",
    )

    def __init__(
        self,
        pca_params: PromptPCAParams,
        max_entries: int = 10_000,
        initial_wnd: int = 8,
        max_wnd: int = 28,
        min_wnd: int = 2,
    ):
        self.pca_params = pca_params
        self.max_entries = max_entries
        K = pca_params.n_components

        # Pre-allocate contiguous arrays (compacted after build)
        self.keys = np.empty((max_entries, K), dtype=np.float16)
        self.entry_rollout_idx = np.empty(max_entries, dtype=np.int32)
        self.entry_offset = np.empty(max_entries, dtype=np.int32)
        self.rewards = np.empty(max_entries, dtype=np.float32)
        self.rollout_seqs: List[np.ndarray] = []
        self.n_entries = 0

        # Adaptive window control
        self.wnd_size = initial_wnd
        self.max_wnd = max_wnd
        self.min_wnd = min_wnd

    # build

    def add_rollout(
        self,
        projected_keys: np.ndarray,
        token_sequence: np.ndarray,
        reward: float = 0.0,
    ) -> int:
        """Append entries from one rollout with **pre-projected** keys.

        Args:
            projected_keys:  (L, K)  float – output of PromptPCAParams.project()
            token_sequence:  (L,)    int32 – response tokens y[0 .. L-1]
            reward:          scalar reward for this rollout

        Returns:
            Number of entries actually written (may be < L when table is full).
        """
        L = len(token_sequence)
        room = self.max_entries - self.n_entries
        if room <= 0 or L == 0:
            return 0
        n_add = min(L, room)

        # Store the full token sequence *once*  (reference-based value)
        ridx = len(self.rollout_seqs)
        self.rollout_seqs.append(
            np.ascontiguousarray(token_sequence[:L], dtype=np.int32)
        )

        # L2-normalise projected keys for cosine similarity
        kf = projected_keys[:n_add].astype(np.float32, copy=False)
        norms = np.linalg.norm(kf, axis=1, keepdims=True)
        np.maximum(norms, 1e-8, out=norms)
        kf /= norms

        s = self.n_entries
        e = s + n_add
        self.keys[s:e] = kf.astype(np.float16)
        self.entry_rollout_idx[s:e] = ridx
        self.entry_offset[s:e] = np.arange(n_add, dtype=np.int32)
        self.rewards[s:e] = reward
        self.n_entries = e
        return n_add

    def compact(self):
        """Shrink pre-allocated arrays to populated size (saves memory)."""
        n = self.n_entries
        if n < self.max_entries:
            self.keys = np.ascontiguousarray(self.keys[:n])
            self.entry_rollout_idx = np.ascontiguousarray(
                self.entry_rollout_idx[:n]
            )
            self.entry_offset = np.ascontiguousarray(self.entry_offset[:n])
            self.rewards = np.ascontiguousarray(self.rewards[:n])
            self.max_entries = n

    # query

    def query(
        self,
        query_z_normalised: np.ndarray,
        threshold: float,
        accept_length: int = 1,
    ) -> Tuple[List[int], float]:
        """Find best match and return draft tokens.

        Args:
            query_z_normalised: (K,) L2-normalised PCA-projected query.
            threshold:          cosine-similarity threshold.
            accept_length:      previous accept length (for window control).

        Returns:
            (draft_tokens, best_similarity).
        """
        if self.n_entries == 0:
            return [], 0.0
        self._update_wnd(accept_length)

        # Dot-product similarity (both sides L2-normalised → cosine sim)
        # Upcast fp16 keys to fp32 for numerical accuracy of the dot.
        sims = self.keys[: self.n_entries].astype(np.float32).dot(
            query_z_normalised.astype(np.float32)
        )  # (M,)

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= threshold:
            draft = self.get_draft_tokens(best_idx, self.wnd_size)
            return draft, best_sim
        return [], best_sim

    def get_draft_tokens(self, entry_idx: int, max_tokens: int) -> List[int]:
        """Retrieve draft tokens via reference  (O(1) slice)."""
        if entry_idx < 0 or entry_idx >= self.n_entries:
            return []
        ridx = int(self.entry_rollout_idx[entry_idx])
        off = int(self.entry_offset[entry_idx])
        seq = self.rollout_seqs[ridx]
        return seq[off : off + max_tokens].tolist()

    # window control

    def _update_wnd(self, accept_length: int):
        if accept_length >= self.wnd_size:
            self.wnd_size = min(self.wnd_size + 1, self.max_wnd)
        elif accept_length <= 1:
            self.wnd_size = max(self.wnd_size // 2, self.min_wnd)

    # accessors

    def get_pca_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, components) as contiguous float32 numpy."""
        return self.pca_params.mean, self.pca_params.components

    def get_keys_numpy(self) -> np.ndarray:
        """Return active keys as contiguous (n_entries, K) float16."""
        return np.ascontiguousarray(self.keys[: self.n_entries])


# Partitioned Ray actor

_num_groups: int = 5


@ray.remote(num_cpus=1)
class HSpecTableGroup:
    """Ray actor managing HSpec tables for one *partition* of prompts.
    """

    def __init__(
        self,
        port: int = 6555,
        similarity_threshold: float = 0.9,
        max_entries_per_prompt: int = 10_000,
        n_components: int = 64,
    ):
        self._tables: Dict[str, PromptTableData] = {}
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries_per_prompt
        self.n_components = n_components
        self.port = port

        # Metrics
        self._query_count = 0
        self._match_count = 0
        self._total_draft_len = 0
        self._build_count = 0
        self._discard_count = 0

        # ZMQ state
        self.running = False

    # Build

    def build_prompt_table(
        self,
        prompt_id: str,
        hidden_states_list: List[np.ndarray],
        token_seq_list: List[Any],
        rewards: List[float],
    ):
        """Build a complete table for one prompt.

        Pipeline:  validate → PCA fit → project → store entries.

        Args:
            prompt_id:          Stable prompt identifier.
            hidden_states_list: [(L_i, D) ndarray] per rollout.
            token_seq_list:     [list[int] | ndarray] per rollout.
            rewards:            [float] per rollout.
        """
        # ① Validate & filter  (design-doc §7: hard alignment check)
        valid_hs: List[np.ndarray] = []
        valid_tok: List[Any] = []
        valid_rew: List[float] = []
        for hs, tok, rew in zip(hidden_states_list, token_seq_list, rewards):
            if hs is None or len(tok) == 0:
                self._discard_count += 1
                continue
            if hs.ndim != 2:
                self._discard_count += 1
                continue
            if hs.shape[0] != len(tok):
                logger.warning(
                    "HSpec alignment mismatch for %s: hs=%d vs tok=%d – "
                    "discarding this trajectory",
                    prompt_id,
                    hs.shape[0],
                    len(tok),
                )
                self._discard_count += 1
                continue
            valid_hs.append(
                hs if hs.dtype == np.float32 else hs.astype(np.float32)
            )
            valid_tok.append(tok)
            valid_rew.append(rew)

        if not valid_hs:
            return

        # PCA fit  (single-sequence for PPO, multi for GRPO)
        K = self.n_components
        try:
            if len(valid_hs) == 1:
                pca_params, proj = fit_pca_single_sequence(
                    prompt_id, valid_hs[0], K
                )
                proj_list = [proj]
            else:
                pca_params, proj_list = fit_pca_multi_sequence(
                    prompt_id, valid_hs, K
                )
        except Exception as exc:
            logger.warning("HSpec PCA failed for %s: %s", prompt_id, exc)
            self._discard_count += len(valid_hs)
            return

        # Create table & populate with projected keys + token refs
        table = PromptTableData(
            pca_params=pca_params, max_entries=self.max_entries
        )
        for proj, tok, rew in zip(proj_list, valid_tok, valid_rew):
            tok_arr = (
                np.asarray(tok, dtype=np.int32)
                if not isinstance(tok, np.ndarray)
                else tok.astype(np.int32, copy=False)
            )
            table.add_rollout(proj, tok_arr, rew)

        table.compact()
        self._tables[prompt_id] = table
        self._build_count += 1

    def build_tables_batch(self, prompt_data_dict: Dict[str, Dict]):
        """Build tables for a *batch* of prompts (one remote call per partition).

        Args:
            prompt_data_dict: ``{prompt_id: {
                'hidden_states': List[ndarray (L_i, D)],
                'tokens':        List[List[int]],
                'rewards':       List[float],
            }}``
        """
        for prompt_id, data in prompt_data_dict.items():
            self.build_prompt_table(
                prompt_id,
                data["hidden_states"],
                data["tokens"],
                data["rewards"],
            )

    # Query

    def query(
        self,
        prompt_id: str,
        hidden_state: np.ndarray,
        accept_length: int = 1,
    ) -> List[int]:
        """Query with a *raw* (D-dim) hidden state.

        Internally projects via the prompt's PCA params, then matches
        against the stored PCA-projected keys.
        """
        self._query_count += 1
        if prompt_id not in self._tables:
            return []

        table = self._tables[prompt_id]
        # Project  → (K,) and L2-normalise
        z = table.pca_params.project(
            hidden_state.reshape(1, -1).astype(np.float32, copy=False)
        ).squeeze(0)  # (K,)
        norm = float(np.linalg.norm(z))
        if norm > 1e-8:
            z /= norm

        draft, sim = table.query(z, self.similarity_threshold, accept_length)
        if draft:
            self._match_count += 1
            self._total_draft_len += len(draft)
        return draft

    def query_batch(
        self,
        prompt_id_list: List[str],
        hidden_state_list: List[np.ndarray],
        accept_length_list: List[int],
    ) -> List[List[int]]:
        """Batch query for multiple prompts."""
        return [
            self.query(pid, hs, al)
            for pid, hs, al in zip(
                prompt_id_list, hidden_state_list, accept_length_list
            )
        ]

    # ── Table data access  (for proposer prefetch / cache) ──

    def get_prompt_pca(self, prompt_id: str):
        """Return ``(mean, components)`` numpy arrays, or ``None``."""
        if prompt_id not in self._tables:
            return None
        t = self._tables[prompt_id]
        return (t.pca_params.mean.copy(), t.pca_params.components.copy())

    def get_prompt_keys(self, prompt_id: str):
        """Return keys ndarray ``(n_entries, K)`` for prompt, or ``None``."""
        if prompt_id not in self._tables:
            return None
        return self._tables[prompt_id].get_keys_numpy()

    def get_prompt_table_data(self, prompt_id: str):
        """Return serialisable dict with full table data for proposer cache."""
        if prompt_id not in self._tables:
            return None
        t = self._tables[prompt_id]
        return {
            "mean": t.pca_params.mean,
            "components": t.pca_params.components,
            "keys": t.get_keys_numpy(),
            "rollout_seqs": [s.tolist() for s in t.rollout_seqs],
            "entry_rollout_idx": np.ascontiguousarray(
                t.entry_rollout_idx[: t.n_entries]
            ),
            "entry_offset": np.ascontiguousarray(
                t.entry_offset[: t.n_entries]
            ),
            "n_entries": t.n_entries,
            "wnd_size": t.wnd_size,
        }

    # Management

    def delete(self, prompt_id: str):
        """Delete a prompt's table."""
        self._tables.pop(prompt_id, None)

    def clear(self):
        """Clear all tables and reset metrics."""
        self._tables.clear()
        self._reset_metrics()

    def exist(self, prompt_id: str) -> bool:
        return prompt_id in self._tables

    def get_prompt_ids(self) -> List[str]:
        return list(self._tables.keys())

    def num_prompts(self) -> int:
        return len(self._tables)

    def total_entries(self) -> int:
        return sum(t.n_entries for t in self._tables.values())

    # Metrics

    def compute_metrics(self) -> Dict[str, float]:
        return {
            "query_times": self._query_count,
            "match_times": self._match_count,
            "total_draft_length": self._total_draft_len,
            "build_count": self._build_count,
            "discard_count": self._discard_count,
            "num_prompts": len(self._tables),
            "total_entries": self.total_entries(),
        }

    def _reset_metrics(self):
        self._query_count = 0
        self._match_count = 0
        self._total_draft_len = 0
        self._build_count = 0
        self._discard_count = 0

    # ZMQ server  (for decode hot-loop queries)

    def run(self):
        """Run blocking ZMQ REP server (call via ``actor.run.remote()``)."""
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://*:{self.port}")
        self.running = True
        logger.info("HSpecTableGroup ZMQ server started on port %d", self.port)

        while self.running:
            try:
                msg = sock.recv()
                req = msgpack.unpackb(msg, raw=False)
                resp = self._handle_zmq(req)
                sock.send(msgpack.packb(resp, use_bin_type=True))
            except Exception as exc:
                try:
                    sock.send(
                        msgpack.packb(
                            {"status": "error", "message": str(exc)},
                            use_bin_type=True,
                        )
                    )
                except Exception:
                    pass

    def _handle_zmq(self, request: Dict) -> Any:
        method = request.get("method")
        params = request.get("params", {})

        if method == "query":
            hs = np.array(params["hidden_state"], dtype=np.float32)
            return self.query(
                params["prompt_id"], hs, params.get("accept_length", 1)
            )
        elif method == "query_batch":
            hs_list = [
                np.array(h, dtype=np.float32)
                for h in params["hidden_state_list"]
            ]
            return self.query_batch(
                params["prompt_id_list"],
                hs_list,
                params["accept_length_list"],
            )
        elif method == "get_prompt_pca":
            result = self.get_prompt_pca(params["prompt_id"])
            if result is None:
                return None
            mean, comp = result
            return {"mean": mean.tolist(), "components": comp.tolist()}
        elif method == "stop":
            self.running = False
            return True
        else:
            return {"error": f"Unknown method: {method}"}


# Global client  (partition routing + async build interface)


class GlobalHSpecTableGroup:
    """Client-side interface for distributed HSpec tables.

    Routes operations to the correct partition actor via
    ``hash(prompt_id) % N``.  Provides async build and batch query
    interfaces for high throughput.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.9,
        max_entries_per_prompt: int = 10_000,
        n_components: int = 64,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries_per_prompt
        self.n_components = n_components

        # Discover existing Ray actors
        self.groups: List[ray.actor.ActorHandle] = []
        for i in range(_num_groups):
            try:
                self.groups.append(ray.get_actor(f"hspec_table_{i}"))
            except ValueError:
                logger.warning("HSpec actor hspec_table_%d not found", i)

        # ZMQ connections (lazy-initialised on first query)
        self._zmq_ctx: Optional[zmq.Context] = None
        self._zmq_sockets: Dict[int, zmq.Socket] = {}

        if self.groups:
            logger.info(
                "HSpec: GlobalHSpecTableGroup connected to %d actors",
                len(self.groups),
            )

    def __len__(self):
        return len(self.groups)

    def _get_partition_id(self, prompt_id: str) -> int:
        return stable_partition_id(prompt_id, _num_groups)

    def _get_partition(self, prompt_id: str) -> ray.actor.ActorHandle:
        return self.groups[self._get_partition_id(prompt_id)]

    # Build  (async, non-blocking)

    def build_tables_async(
        self, prompt_data: Dict[str, Dict]
    ) -> List[ray.ObjectRef]:
        """Send rollout data to partition actors for async PCA fitting + build.

        This is the main Step 1 entry point called by the trainer.
        PCA fitting + table construction runs inside Ray actors and does
        **not** block the caller.

        Args:
            prompt_data: ``{prompt_id: {
                'hidden_states': List[ndarray (L_i, D)],
                'tokens':        List[List[int]],
                'rewards':       List[float],
            }}``

        Returns:
            Ray ObjectRefs (futures).  Call ``ray.get(refs)`` only when you
            need to guarantee building is complete (e.g. before starting
            the next rollout that will query the tables).
        """
        # Group by partition
        partition_payloads: Dict[int, Dict[str, Dict]] = {
            i: {} for i in range(_num_groups)
        }
        for prompt_id, data in prompt_data.items():
            pid = self._get_partition_id(prompt_id)
            partition_payloads[pid][prompt_id] = data

        futures: List[ray.ObjectRef] = []
        for pid, payload in partition_payloads.items():
            if payload and pid < len(self.groups):
                futures.append(
                    self.groups[pid].build_tables_batch.remote(payload)
                )
        return futures

    # Query

    def query(self, prompt_id: str, hidden_state: np.ndarray,
              accept_length: int = 1):
        """Single query via Ray actor (returns ObjectRef / future)."""
        actor = self._get_partition(prompt_id)
        return actor.query.remote(prompt_id, hidden_state, accept_length)

    def query_batch(
        self,
        prompt_id_list: List[str],
        hidden_state_list: List[np.ndarray],
        accept_length_list: List[int],
    ) -> List[List[int]]:
        """Batch query via Ray actors (blocking – collects results)."""
        parts: Dict[int, Dict[str, list]] = {
            i: {"pids": [], "hss": [], "als": [], "pos": []}
            for i in range(_num_groups)
        }
        for idx, (pid, hs, al) in enumerate(
            zip(prompt_id_list, hidden_state_list, accept_length_list)
        ):
            p = self._get_partition_id(pid)
            parts[p]["pids"].append(pid)
            parts[p]["hss"].append(hs)
            parts[p]["als"].append(al)
            parts[p]["pos"].append(idx)

        futures = {}
        for p, d in parts.items():
            if d["pids"] and p < len(self.groups):
                futures[p] = self.groups[p].query_batch.remote(
                    d["pids"], d["hss"], d["als"]
                )

        results: List[List[int]] = [[] for _ in range(len(prompt_id_list))]
        for p, future in futures.items():
            batch_res = ray.get(future)
            for pos, draft in zip(parts[p]["pos"], batch_res):
                results[pos] = draft
        return results

    def post_query_batch(
        self,
        prompt_id_list: List[str],
        hidden_state_list: List[np.ndarray],
        accept_length_list: List[int],
    ) -> List[List[int]]:
        """Batch query via ZMQ  (lower latency for decode hot path).

        Hidden states are sent as raw D-dim vectors; the table actor
        projects them internally via PCA.  Step 2 will optimise this by
        sending pre-projected K-dim vectors from on-device cache.
        """
        self._ensure_zmq()

        parts: Dict[int, Dict[str, list]] = {
            i: {"pids": [], "hss": [], "als": [], "pos": []}
            for i in range(_num_groups)
        }
        for idx, (pid, hs, al) in enumerate(
            zip(prompt_id_list, hidden_state_list, accept_length_list)
        ):
            p = self._get_partition_id(pid)
            parts[p]["pids"].append(pid)
            parts[p]["hss"].append(
                hs.tolist() if isinstance(hs, np.ndarray) else hs
            )
            parts[p]["als"].append(al)
            parts[p]["pos"].append(idx)

        responses: Dict[int, Any] = {}
        for sid, d in parts.items():
            if d["pids"] and sid in self._zmq_sockets:
                req = {
                    "method": "query_batch",
                    "params": {
                        "prompt_id_list": d["pids"],
                        "hidden_state_list": d["hss"],
                        "accept_length_list": d["als"],
                    },
                }
                responses[sid] = self._zmq_send(sid, req)

        results: List[List[int]] = [[] for _ in range(len(prompt_id_list))]
        for sid, resp in responses.items():
            if isinstance(resp, list):
                for pos, draft in zip(parts[sid]["pos"], resp):
                    results[pos] = draft
        return results

    # Table data access  (for proposer prefetch / cache)

    def get_prompt_pca(self, prompt_id: str):
        """Return future resolving to ``(mean, components)`` or ``None``."""
        return self._get_partition(prompt_id).get_prompt_pca.remote(prompt_id)

    def get_prompt_table_data(self, prompt_id: str):
        """Return future resolving to serialisable table data dict."""
        return self._get_partition(prompt_id).get_prompt_table_data.remote(
            prompt_id
        )

    def get_prompt_table_data_batch(
        self, prompt_id_list: List[str]
    ) -> Dict[str, ray.ObjectRef]:
        """Batch fetch table data  (returns ``{prompt_id: future}``)."""
        futures: Dict[str, ray.ObjectRef] = {}
        for pid in prompt_id_list:
            p = self._get_partition_id(pid)
            if p < len(self.groups):
                futures[pid] = self.groups[
                    p
                ].get_prompt_table_data.remote(pid)
        return futures

    # Management

    def clear(self):
        """Clear all tables (returns list of futures)."""
        return [g.clear.remote() for g in self.groups]

    def delete(self, prompt_id: str):
        return self._get_partition(prompt_id).delete.remote(prompt_id)

    # Metrics

    def compute_metrics(self) -> Dict[str, float]:
        """Aggregate metrics from all partition actors."""
        if not self.groups:
            return {
                "hspec/match_rate": 0.0,
                "hspec/avg_draft_length": 0.0,
                "hspec/query_times": 0,
                "hspec/match_times": 0,
                "hspec/build_count": 0,
                "hspec/discard_count": 0,
                "hspec/num_prompts": 0,
                "hspec/total_entries": 0,
            }
        tasks = [g.compute_metrics.remote() for g in self.groups]
        metrics_list = ray.get(tasks)

        agg: Dict[str, float] = {}
        for key in metrics_list[0]:
            agg[key] = sum(float(m[key]) for m in metrics_list)

        qt = agg.get("query_times", 0)
        mt = agg.get("match_times", 0)
        tdl = agg.get("total_draft_length", 0)
        return {
            "hspec/match_rate": mt / qt if qt > 0 else 0.0,
            "hspec/avg_draft_length": tdl / mt if mt > 0 else 0.0,
            "hspec/query_times": qt,
            "hspec/match_times": mt,
            "hspec/build_count": agg.get("build_count", 0),
            "hspec/discard_count": agg.get("discard_count", 0),
            "hspec/num_prompts": agg.get("num_prompts", 0),
            "hspec/total_entries": agg.get("total_entries", 0),
        }

    # ZMQ helpers

    def _ensure_zmq(self):
        if self._zmq_sockets:
            return
        self._zmq_ctx = zmq.Context()
        for i in range(_num_groups):
            sock = self._zmq_ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 5000)
            sock.connect(f"tcp://localhost:{6555 + i}")
            self._zmq_sockets[i] = sock

    def _zmq_send(self, server_id: int, request: Dict) -> Any:
        sock = self._zmq_sockets.get(server_id)
        if sock is None:
            return []
        try:
            sock.send(msgpack.packb(request, use_bin_type=True))
            resp = sock.recv()
            return msgpack.unpackb(resp, raw=False)
        except Exception:
            return []

    def run_server(self):
        """Start ZMQ servers on all actors (non-blocking futures)."""
        return [g.run.remote() for g in self.groups]

    def stop_server(self):
        """Send stop command to all ZMQ servers."""
        self._ensure_zmq()
        for sid in list(self._zmq_sockets):
            self._zmq_send(sid, {"method": "stop", "params": {}})


# Init / get  (module-level helpers)

_hspec_table_handles: List = []


def init_hspec_tables(
    similarity_threshold: float = 0.9,
    max_entries_per_prompt: int = 10_000,
    n_components: int = 64,
):
    """Create and register HSpec Ray actors.  Call once at training start."""
    global _hspec_table_handles
    _hspec_table_handles = []

    for i in range(_num_groups):
        handle = HSpecTableGroup.options(name=f"hspec_table_{i}").remote(
            port=6555 + i,
            similarity_threshold=similarity_threshold,
            max_entries_per_prompt=max_entries_per_prompt,
            n_components=n_components,
        )
        _hspec_table_handles.append(handle)

    for i in range(_num_groups):
        try:
            ray.get_actor(f"hspec_table_{i}")
            logger.info("HSpec Actor %d registered successfully.", i)
        except ValueError:
            logger.error("HSpec Actor %d failed to register!", i)


def get_hspec_tables(
    similarity_threshold: float = 0.9,
    max_entries_per_prompt: int = 10_000,
    n_components: int = 64,
) -> GlobalHSpecTableGroup:
    """Get or create the global HSpec table manager.

    If actors do not exist yet they are created automatically.
    """
    # Ensure actors exist
    needs_init = False
    for i in range(_num_groups):
        try:
            ray.get_actor(f"hspec_table_{i}")
        except ValueError:
            needs_init = True
            break

    if needs_init:
        init_hspec_tables(
            similarity_threshold, max_entries_per_prompt, n_components
        )

    return GlobalHSpecTableGroup(
        similarity_threshold=similarity_threshold,
        max_entries_per_prompt=max_entries_per_prompt,
        n_components=n_components,
    )
