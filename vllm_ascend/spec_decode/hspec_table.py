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
HSpec: Hidden State based Speculative Decoding for RL Training.

This module implements the query table data structure for HSpec,
which stores hidden_state -> remaining_tokens mappings for each prompt.
"""

import ray
import zmq
import math
import msgpack
import numpy as np
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any
import torch


class HSpecEntry:
    """Single entry in the HSpec query table.
    
    Attributes:
        hidden_state: The hidden state vector (normalized).
        remaining_tokens: The remaining token sequence after this position.
        reward: The cumulative reward score for this sequence.
    """
    
    def __init__(self, hidden_state: np.ndarray, remaining_tokens: List[int], reward: float = 0.0):
        self.hidden_state = hidden_state  # Shape: (hidden_dim,)
        self.remaining_tokens = remaining_tokens  # List of token ids
        self.reward = reward
    
    def get_draft_tokens(self, max_tokens: int) -> List[int]:
        """Get up to max_tokens draft tokens from remaining sequence."""
        return self.remaining_tokens[:max_tokens]


class HSpecQueryTable:
    """Query table for a single prompt that stores hidden_state -> remaining_tokens mappings.
    
    The table supports efficient similarity search using cosine similarity.
    """
    
    def __init__(self, similarity_threshold: float = 0.9, max_entries: int = 10000):
        """Initialize the query table.
        
        Args:
            similarity_threshold: Minimum cosine similarity for a match.
            max_entries: Maximum number of entries to store.
        """
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.entries: List[HSpecEntry] = []
        
        # For batch similarity computation
        self._hidden_states_matrix: Optional[np.ndarray] = None
        self._matrix_dirty = True
        
        # Adaptive window control (similar to HistoSpec)
        self.wnd_size: int = 8
        self.ssthresh = 16
        self.max_wnd = 28
        self.min_wnd = 2
    
    def add_entry(self, hidden_state: np.ndarray, remaining_tokens: List[int], reward: float = 0.0):
        """Add a new entry to the query table.
        
        Args:
            hidden_state: Hidden state vector from the model.
            remaining_tokens: Remaining token sequence after this position.
            reward: Reward score for this sequence.
        """
        if len(self.entries) >= self.max_entries:
            # TODO: Implement eviction strategy (e.g., remove lowest reward entries)
            return
        
        # Normalize hidden state for cosine similarity
        norm = np.linalg.norm(hidden_state)
        if norm > 0:
            hidden_state = hidden_state / norm
        
        entry = HSpecEntry(hidden_state, remaining_tokens, reward)
        self.entries.append(entry)
        self._matrix_dirty = True
    
    def _build_matrix(self):
        """Build the hidden states matrix for batch similarity computation."""
        if not self._matrix_dirty or len(self.entries) == 0:
            return
        
        self._hidden_states_matrix = np.stack(
            [entry.hidden_state for entry in self.entries], axis=0
        )  # Shape: (num_entries, hidden_dim)
        self._matrix_dirty = False
    
    def query(self, query_hidden_state: np.ndarray, accept_length: int = 1) -> Tuple[List[int], float]:
        """Query the table with a hidden state vector.
        
        Args:
            query_hidden_state: The query hidden state vector.
            accept_length: Number of tokens accepted in the previous iteration (for window adjustment).
        
        Returns:
            Tuple of (draft_tokens, similarity_score).
            Returns empty list if no match found.
        """
        if len(self.entries) == 0:
            return [], 0.0
        
        # Update window size based on accept_length
        self._update_window_size(accept_length)
        
        # Normalize query vector
        norm = np.linalg.norm(query_hidden_state)
        if norm > 0:
            query_hidden_state = query_hidden_state / norm
        
        # Build matrix if needed
        self._build_matrix()
        
        # Compute cosine similarities (dot product since vectors are normalized)
        similarities = np.dot(self._hidden_states_matrix, query_hidden_state)
        
        # Find best match above threshold
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= self.similarity_threshold:
            draft_tokens = self.entries[best_idx].get_draft_tokens(self.wnd_size)
            return draft_tokens, best_similarity
        
        return [], best_similarity
    
    def _update_window_size(self, accept_length: int):
        """Update the prediction window size based on acceptance rate."""
        # Similar to HistoSpec's congestion control
        if accept_length >= self.wnd_size:
            # Good prediction, increase window
            self.wnd_size = min(self.wnd_size + 1, self.max_wnd)
        elif accept_length <= 1:
            # Poor prediction, decrease window
            self.wnd_size = max(self.wnd_size // 2, self.min_wnd)
    
    def clear(self):
        """Clear all entries from the table."""
        self.entries.clear()
        self._hidden_states_matrix = None
        self._matrix_dirty = True
        self.wnd_size = 8


@ray.remote(num_cpus=1)
class HSpecTableGroup:
    """Ray Actor that manages HSpec query tables for multiple prompts.
    
    This is the distributed storage component for HSpec, similar to
    SuffixTreeGroup in HistoSpec.
    """
    
    def __init__(self, port: int = 6555, similarity_threshold: float = 0.9):
        """Initialize the table group.
        
        Args:
            port: ZMQ server port for fast RPC communication.
            similarity_threshold: Default similarity threshold for tables.
        """
        self._tables: Dict[str, HSpecQueryTable] = {}
        self.similarity_threshold = similarity_threshold
        self.port = port
        
        # Metrics
        self.query_times = 0
        self.match_times = 0
        self.total_draft_length = 0
        
        # Server state
        self.running = False
    
    def __len__(self):
        return len(self._tables)
    
    def add_table(self, prompt_id: str):
        """Create a new query table for a prompt."""
        if prompt_id not in self._tables:
            self._tables[prompt_id] = HSpecQueryTable(
                similarity_threshold=self.similarity_threshold
            )
    
    def add_entry(self, prompt_id: str, hidden_state: np.ndarray, 
                  remaining_tokens: List[int], reward: float = 0.0):
        """Add an entry to a prompt's query table.
        
        Args:
            prompt_id: The prompt identifier.
            hidden_state: Hidden state vector.
            remaining_tokens: Remaining token sequence.
            reward: Reward score.
        """
        if prompt_id not in self._tables:
            raise ValueError(f"{prompt_id} not in HSpecTableGroup")
        self._tables[prompt_id].add_entry(hidden_state, remaining_tokens, reward)
    
    def add_entries_batch(self, prompt_id: str, hidden_states: np.ndarray,
                          token_sequence: List[int], reward: float = 0.0):
        """Add multiple entries from a single rollout sequence.
        
        This is the main method called after rollout to populate the table.
        For each position i in the sequence, it stores:
            hidden_state[i] -> token_sequence[i+1:]
        
        Args:
            prompt_id: The prompt identifier.
            hidden_states: Hidden states for all positions. Shape: (seq_len, hidden_dim)
            token_sequence: The complete rollout token sequence.
            reward: Total reward for this sequence.
        """
        if prompt_id not in self._tables:
            raise ValueError(f"{prompt_id} not in HSpecTableGroup")
        
        table = self._tables[prompt_id]
        seq_len = len(token_sequence)
        
        # For each position, store hidden_state -> remaining tokens
        for i in range(min(len(hidden_states), seq_len - 1)):
            remaining = token_sequence[i + 1:]
            if len(remaining) > 0:
                table.add_entry(hidden_states[i], remaining, reward)
    
    def query(self, prompt_id: str, hidden_state: np.ndarray, 
              accept_length: int = 1) -> List[int]:
        """Query for draft tokens given a hidden state.
        
        Args:
            prompt_id: The prompt identifier.
            hidden_state: Current hidden state from the model.
            accept_length: Number of tokens accepted in previous iteration.
        
        Returns:
            List of draft token ids, or empty list if no match.
        """
        self.query_times += 1
        
        if prompt_id not in self._tables:
            return []
        
        draft_tokens, similarity = self._tables[prompt_id].query(hidden_state, accept_length)
        
        if len(draft_tokens) > 0:
            self.match_times += 1
            self.total_draft_length += len(draft_tokens)
        
        return draft_tokens
    
    def query_batch(self, prompt_id_list: List[str], hidden_state_list: List[np.ndarray],
                    accept_length_list: List[int]) -> List[List[int]]:
        """Batch query for multiple prompts.
        
        Args:
            prompt_id_list: List of prompt identifiers.
            hidden_state_list: List of hidden state vectors.
            accept_length_list: List of accept lengths.
        
        Returns:
            List of draft token lists for each query.
        """
        results = []
        for prompt_id, hidden_state, accept_length in zip(
            prompt_id_list, hidden_state_list, accept_length_list
        ):
            draft_tokens = self.query(prompt_id, hidden_state, accept_length)
            results.append(draft_tokens)
        return results
    
    def delete(self, prompt_id: str):
        """Delete a prompt's query table."""
        if prompt_id in self._tables:
            del self._tables[prompt_id]
    
    def exist(self, prompt_id: str) -> bool:
        """Check if a prompt's table exists."""
        return prompt_id in self._tables
    
    def get_prompt_ids(self) -> List[str]:
        """Get all prompt ids in this group."""
        return list(self._tables.keys())
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute and return metrics."""
        return {
            'query_times': self.query_times,
            'match_times': self.match_times,
            'total_draft_length': self.total_draft_length,
        }
    
    def clear(self):
        """Clear all tables and reset metrics."""
        self._tables.clear()
        self.query_times = 0
        self.match_times = 0
        self.total_draft_length = 0
    
    # ==================== ZMQ Server Methods ====================
    
    def run(self):
        """Run the ZMQ server for fast RPC communication."""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.port}")
        
        self.running = True
        print(f"HSpecTableGroup server started on port {self.port}")
        
        while self.running:
            try:
                message = socket.recv()
                request = msgpack.unpackb(message, raw=False)
                response = self._handle_request(request)
                socket.send(msgpack.packb(response, use_bin_type=True))
            except Exception as e:
                error_response = {'status': 'error', 'message': str(e)}
                socket.send(msgpack.packb(error_response, use_bin_type=True))
    
    def _handle_request(self, request: Dict[str, Any]) -> Any:
        """Handle incoming RPC request."""
        method = request.get('method')
        params = request.get('params', {})
        
        if method == 'query':
            hidden_state = np.array(params['hidden_state'], dtype=np.float32)
            return self.query(
                params['prompt_id'],
                hidden_state,
                params.get('accept_length', 1)
            )
        elif method == 'query_batch':
            hidden_states = [np.array(hs, dtype=np.float32) for hs in params['hidden_state_list']]
            return self.query_batch(
                params['prompt_id_list'],
                hidden_states,
                params['accept_length_list']
            )
        elif method == 'stop':
            self.running = False
            return True
        else:
            return {'error': f'Unknown method: {method}'}


# ==================== Global HSpec Table Manager ====================

_num_groups: int = 5  # Number of partitions (same as HistoSpec)
_hspec_table_handles = []


class GlobalHSpecTableGroup:
    """Global manager for distributed HSpec query tables.
    
    All functions return Ray futures for non-blocking operation.
    Similar to GlobalRewardAwareSuffixTreeGroup in HistoSpec.
    """
    
    def __init__(self, similarity_threshold: float = 0.9):
        """Initialize the global table group.
        
        Args:
            similarity_threshold: Default similarity threshold for matching.
        """
        self.groups: List[ray.actor.ActorHandle] = []
        self.similarity_threshold = similarity_threshold
        
        # Try to get existing actors
        for i in range(_num_groups):
            try:
                actor_handle = ray.get_actor(f"hspec_table_{i}")
                self.groups.append(actor_handle)
            except ValueError:
                print(f"Could not find HSpec actor {i}")
        
        # Setup ZMQ connections for fast queries
        self.server_configs = {}
        for i in range(_num_groups):
            self.server_configs[i] = {
                'host': 'localhost',
                'port': 6555 + i,
            }
        self._setup_connections()
    
    def _setup_connections(self):
        """Setup ZMQ connections to all table groups."""
        self.servers = {}
        for server_id, config in self.server_configs.items():
            socket = zmq.Context().socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, config.get('timeout', 5000))
            server_url = f"tcp://{config['host']}:{config['port']}"
            socket.connect(server_url)
            self.servers[server_id] = {
                'socket': socket,
                'config': config,
                'url': server_url
            }
        print(f"HSpec: Connected to {len(self.servers)} servers")
    
    def __len__(self):
        return len(self.groups)
    
    def _get_partition_id(self, prompt_id: str) -> int:
        """Get the partition id for a prompt."""
        return hash(prompt_id) % _num_groups
    
    def _get_partition(self, prompt_id: str) -> ray.actor.ActorHandle:
        """Get the Ray actor for a prompt."""
        group_index = self._get_partition_id(prompt_id)
        return self.groups[group_index]
    
    # ==================== Table Management ====================
    
    def add_table(self, prompt_id: str):
        """Create a new query table for a prompt."""
        actor = self._get_partition(prompt_id)
        return actor.add_table.remote(prompt_id)
    
    def add_entry(self, prompt_id: str, hidden_state: np.ndarray,
                  remaining_tokens: List[int], reward: float = 0.0):
        """Add a single entry to a prompt's table."""
        actor = self._get_partition(prompt_id)
        return actor.add_entry.remote(prompt_id, hidden_state, remaining_tokens, reward)
    
    def add_entries_batch(self, prompt_id: str, hidden_states: np.ndarray,
                          token_sequence: List[int], reward: float = 0.0):
        """Add entries from a complete rollout sequence."""
        actor = self._get_partition(prompt_id)
        return actor.add_entries_batch.remote(prompt_id, hidden_states, token_sequence, reward)
    
    def delete(self, prompt_id: str):
        """Delete a prompt's table."""
        actor = self._get_partition(prompt_id)
        return actor.delete.remote(prompt_id)
    
    def exist(self, prompt_id: str):
        """Check if a prompt's table exists."""
        actor = self._get_partition(prompt_id)
        return actor.exist.remote(prompt_id)
    
    def clear(self):
        """Clear all tables."""
        return [p.clear.remote() for p in self.groups]
    
    # ==================== Query Methods ====================
    
    def query(self, prompt_id: str, hidden_state: np.ndarray, accept_length: int = 1):
        """Query for draft tokens (Ray remote call)."""
        actor = self._get_partition(prompt_id)
        return actor.query.remote(prompt_id, hidden_state, accept_length)
    
    def query_batch(self, prompt_id_list: List[str], hidden_state_list: List[np.ndarray],
                    accept_length_list: List[int]) -> List[List[int]]:
        """Batch query using Ray actors."""
        # Partition queries by group
        params = {i: {
            'prompt_id_list': [],
            'hidden_state_list': [],
            'accept_length_list': [],
            'positions': []  # Track original positions
        } for i in range(_num_groups)}
        
        for idx, (prompt_id, hidden_state, accept_length) in enumerate(
            zip(prompt_id_list, hidden_state_list, accept_length_list)
        ):
            partition_id = self._get_partition_id(prompt_id)
            params[partition_id]['prompt_id_list'].append(prompt_id)
            params[partition_id]['hidden_state_list'].append(hidden_state)
            params[partition_id]['accept_length_list'].append(accept_length)
            params[partition_id]['positions'].append(idx)
        
        # Send batch queries to each group
        futures = {}
        for i in range(_num_groups):
            if params[i]['prompt_id_list']:
                futures[i] = self.groups[i].query_batch.remote(
                    params[i]['prompt_id_list'],
                    params[i]['hidden_state_list'],
                    params[i]['accept_length_list']
                )
        
        # Collect results
        results = [[] for _ in range(len(prompt_id_list))]
        for i, future in futures.items():
            group_results = ray.get(future)
            for pos, draft_tokens in zip(params[i]['positions'], group_results):
                results[pos] = draft_tokens
        
        return results
    
    def post_query_batch(self, prompt_id_list: List[str], hidden_state_list: List[np.ndarray],
                         accept_length_list: List[int]) -> List[List[int]]:
        """Batch query using ZMQ server (faster for inference)."""
        # Partition queries by group
        params = {i: {
            'prompt_id_list': [],
            'hidden_state_list': [],
            'accept_length_list': [],
            'positions': []
        } for i in range(_num_groups)}
        
        for idx, (prompt_id, hidden_state, accept_length) in enumerate(
            zip(prompt_id_list, hidden_state_list, accept_length_list)
        ):
            partition_id = self._get_partition_id(prompt_id)
            params[partition_id]['prompt_id_list'].append(prompt_id)
            # Convert to list for msgpack serialization
            params[partition_id]['hidden_state_list'].append(hidden_state.tolist())
            params[partition_id]['accept_length_list'].append(accept_length)
            params[partition_id]['positions'].append(idx)
        
        # Send to ZMQ servers
        responses = {}
        for server_id, param in params.items():
            if param['prompt_id_list']:
                request = {
                    'method': 'query_batch',
                    'params': {
                        'prompt_id_list': param['prompt_id_list'],
                        'hidden_state_list': param['hidden_state_list'],
                        'accept_length_list': param['accept_length_list']
                    }
                }
                responses[server_id] = self._post_to_server(server_id, request)
        
        # Reconstruct results in original order
        results = [[] for _ in range(len(prompt_id_list))]
        for server_id, response in responses.items():
            if isinstance(response, list):
                for pos, draft_tokens in zip(params[server_id]['positions'], response):
                    results[pos] = draft_tokens
        
        return results
    
    def _post_to_server(self, server_id: int, request: Dict) -> Any:
        """Send request to ZMQ server."""
        if server_id not in self.servers:
            return {'status': 'error', 'message': f'Unknown server: {server_id}'}
        
        socket = self.servers[server_id]['socket']
        try:
            socket.send(msgpack.packb(request, use_bin_type=True))
            response = socket.recv()
            return msgpack.unpackb(response, raw=False)
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    # ==================== Server Control ====================
    
    def run_server(self):
        """Start ZMQ servers on all groups."""
        return [p.run.remote() for p in self.groups]
    
    def stop_server(self):
        """Stop all ZMQ servers."""
        for server_id in self.servers:
            self._post_to_server(server_id, {'method': 'stop', 'params': {}})
    
    # ==================== Metrics ====================
    
    def compute_metrics(self) -> Dict[str, float]:
        """Aggregate metrics from all groups."""
        tasks = [group.compute_metrics.remote() for group in self.groups]
        metrics_list = ray.get(tasks)
        
        data = {}
        for key in metrics_list[0].keys():
            data[key] = sum(m[key] for m in metrics_list)
        
        match_rate = data['match_times'] / data['query_times'] if data['query_times'] > 0 else 0
        avg_draft_len = data['total_draft_length'] / data['match_times'] if data['match_times'] > 0 else 0
        
        return {
            'hspec/match_rate': match_rate,
            'hspec/avg_draft_length': avg_draft_len,
            'hspec/query_times': data['query_times'],
            'hspec/match_times': data['match_times'],
        }


def init_hspec_tables(similarity_threshold: float = 0.9):
    """Initialize the global HSpec tables as Ray actors.
    
    This should be called once at the start of training.
    
    Args:
        similarity_threshold: Default similarity threshold for matching.
    """
    global _hspec_table_handles
    _hspec_table_handles = []
    
    for i in range(_num_groups):
        actor_handle = HSpecTableGroup.options(
            name=f"hspec_table_{i}"
        ).remote(port=6555 + i, similarity_threshold=similarity_threshold)
        _hspec_table_handles.append(actor_handle)
    
    # Verify actors are registered
    for i in range(_num_groups):
        try:
            ray.get_actor(f"hspec_table_{i}")
            print(f"HSpec Actor {i} registered successfully.")
        except ValueError:
            print(f"HSpec Actor {i} failed to register.")


def get_hspec_tables(similarity_threshold: float = 0.9) -> GlobalHSpecTableGroup:
    """Get the global HSpec table manager.
    
    Args:
        similarity_threshold: Default similarity threshold for matching.
    
    Returns:
        GlobalHSpecTableGroup instance.
    """
    return GlobalHSpecTableGroup(similarity_threshold=similarity_threshold)
