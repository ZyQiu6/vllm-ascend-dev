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
Utility functions for HSpec (Hidden State based Speculative Decoding).

This module provides helper functions for hidden state collection and processing.
"""

from typing import Optional, List, Tuple, Dict, Any
import torch
import numpy as np


class HiddenStateCollector:
    """Collector for hidden states during model inference.
    
    This class helps collect and manage hidden states during generation,
    which are later used to build the HSpec query tables.
    """
    
    def __init__(self, hidden_dim: int, max_seq_len: int = 4096, device: str = "cpu"):
        """Initialize the hidden state collector.
        
        Args:
            hidden_dim: Dimension of hidden states.
            max_seq_len: Maximum sequence length to collect.
            device: Device for tensor operations.
        """
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.device = device
        
        # Storage for collected hidden states
        self._hidden_states: Dict[str, List[np.ndarray]] = {}
        self._token_ids: Dict[str, List[int]] = {}
    
    def start_collection(self, req_id: str):
        """Start collecting hidden states for a request.
        
        Args:
            req_id: The request identifier.
        """
        self._hidden_states[req_id] = []
        self._token_ids[req_id] = []
    
    def collect(self, req_id: str, hidden_state: np.ndarray, token_id: int):
        """Collect a hidden state for a request.
        
        Args:
            req_id: The request identifier.
            hidden_state: The hidden state vector. Shape: (hidden_dim,)
            token_id: The corresponding token id.
        """
        if req_id not in self._hidden_states:
            self.start_collection(req_id)
        
        if len(self._hidden_states[req_id]) < self.max_seq_len:
            self._hidden_states[req_id].append(hidden_state)
            self._token_ids[req_id].append(token_id)
    
    def collect_batch(self, req_id: str, hidden_states: np.ndarray, token_ids: List[int]):
        """Collect a batch of hidden states for a request.
        
        Args:
            req_id: The request identifier.
            hidden_states: Hidden states array. Shape: (seq_len, hidden_dim)
            token_ids: List of token ids.
        """
        if req_id not in self._hidden_states:
            self.start_collection(req_id)
        
        for i, (hs, tid) in enumerate(zip(hidden_states, token_ids)):
            if len(self._hidden_states[req_id]) < self.max_seq_len:
                self._hidden_states[req_id].append(hs)
                self._token_ids[req_id].append(tid)
    
    def get_collected(self, req_id: str) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        """Get collected hidden states and token ids for a request.
        
        Args:
            req_id: The request identifier.
        
        Returns:
            Tuple of (hidden_states array, token_ids list), or (None, None) if not found.
        """
        if req_id not in self._hidden_states:
            return None, None
        
        hidden_states = np.stack(self._hidden_states[req_id], axis=0)
        token_ids = self._token_ids[req_id]
        
        return hidden_states, token_ids
    
    def clear_request(self, req_id: str):
        """Clear collected data for a request.
        
        Args:
            req_id: The request identifier.
        """
        if req_id in self._hidden_states:
            del self._hidden_states[req_id]
        if req_id in self._token_ids:
            del self._token_ids[req_id]
    
    def clear_all(self):
        """Clear all collected data."""
        self._hidden_states.clear()
        self._token_ids.clear()


def extract_last_hidden_state(
    hidden_states: torch.Tensor,
    logits_indices: torch.Tensor,
    device: str = "cpu"
) -> np.ndarray:
    """Extract the last hidden state for each request.
    
    Args:
        hidden_states: Hidden states tensor. Shape: (total_tokens, hidden_dim)
        logits_indices: Indices of the last token for each request.
        device: Device for tensor operations.
    
    Returns:
        NumPy array of last hidden states. Shape: (num_requests, hidden_dim)
    """
    # Index hidden states at logits positions
    last_hidden_states = hidden_states[logits_indices]
    
    # Convert to numpy
    return last_hidden_states.cpu().numpy()


def compute_similarity(
    query: np.ndarray,
    keys: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """Compute cosine similarity between query and keys.
    
    Args:
        query: Query vector. Shape: (hidden_dim,)
        keys: Key matrix. Shape: (num_keys, hidden_dim)
        normalize: Whether to normalize vectors before computing similarity.
    
    Returns:
        Similarity scores. Shape: (num_keys,)
    """
    if normalize:
        # Normalize query
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
        
        # Normalize keys
        keys_norm = np.linalg.norm(keys, axis=1, keepdims=True)
        keys_norm = np.maximum(keys_norm, 1e-8)
        keys = keys / keys_norm
    
    # Compute dot product (cosine similarity since vectors are normalized)
    similarities = np.dot(keys, query)
    
    return similarities


def batch_compute_similarity(
    queries: np.ndarray,
    keys: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """Batch compute cosine similarity between queries and keys.
    
    Args:
        queries: Query matrix. Shape: (num_queries, hidden_dim)
        keys: Key matrix. Shape: (num_keys, hidden_dim)
        normalize: Whether to normalize vectors.
    
    Returns:
        Similarity matrix. Shape: (num_queries, num_keys)
    """
    if normalize:
        # Normalize queries
        queries_norm = np.linalg.norm(queries, axis=1, keepdims=True)
        queries_norm = np.maximum(queries_norm, 1e-8)
        queries = queries / queries_norm
        
        # Normalize keys
        keys_norm = np.linalg.norm(keys, axis=1, keepdims=True)
        keys_norm = np.maximum(keys_norm, 1e-8)
        keys = keys / keys_norm
    
    # Compute similarity matrix
    similarities = np.dot(queries, keys.T)
    
    return similarities


def find_best_match(
    query: np.ndarray,
    keys: np.ndarray,
    threshold: float = 0.9,
    normalize: bool = True
) -> Tuple[int, float]:
    """Find the best matching key for a query.
    
    Args:
        query: Query vector. Shape: (hidden_dim,)
        keys: Key matrix. Shape: (num_keys, hidden_dim)
        threshold: Minimum similarity threshold for a valid match.
        normalize: Whether to normalize vectors.
    
    Returns:
        Tuple of (best_index, best_similarity).
        Returns (-1, 0.0) if no match above threshold.
    """
    similarities = compute_similarity(query, keys, normalize)
    
    best_idx = np.argmax(similarities)
    best_sim = similarities[best_idx]
    
    if best_sim >= threshold:
        return int(best_idx), float(best_sim)
    else:
        return -1, 0.0


def prepare_hidden_states_for_storage(
    hidden_states: torch.Tensor,
    token_ids: List[int],
    pad_token_id: int = 0
) -> Tuple[np.ndarray, List[int]]:
    """Prepare hidden states for storage in HSpec tables.
    
    This function:
    1. Removes padding positions
    2. Converts to numpy
    3. Returns aligned hidden states and token ids
    
    Args:
        hidden_states: Hidden states tensor. Shape: (seq_len, hidden_dim)
        token_ids: List of token ids (may include padding).
        pad_token_id: The padding token id.
    
    Returns:
        Tuple of (hidden_states_np, valid_token_ids).
    """
    # Find valid (non-padding) positions
    valid_positions = [i for i, tid in enumerate(token_ids) if tid != pad_token_id]
    
    if not valid_positions:
        return np.array([]), []
    
    # Extract valid hidden states
    valid_indices = torch.tensor(valid_positions, dtype=torch.long)
    valid_hidden_states = hidden_states[valid_indices].cpu().numpy()
    valid_token_ids = [token_ids[i] for i in valid_positions]
    
    return valid_hidden_states, valid_token_ids


class HSpecConfig:
    """Configuration class for HSpec.
    
    This class holds all configuration parameters for HSpec.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.9,
        num_speculative_tokens: int = 5,
        min_match_len: int = 1,
        max_entries_per_table: int = 10000,
        initial_window_size: int = 8,
        max_window_size: int = 28,
        min_window_size: int = 2,
    ):
        """Initialize HSpec configuration.
        
        Args:
            similarity_threshold: Minimum cosine similarity for a match.
            num_speculative_tokens: Maximum number of draft tokens to propose.
            min_match_len: Minimum sequence length before attempting matching.
            max_entries_per_table: Maximum entries per prompt's query table.
            initial_window_size: Initial prediction window size.
            max_window_size: Maximum window size.
            min_window_size: Minimum window size.
        """
        self.similarity_threshold = similarity_threshold
        self.num_speculative_tokens = num_speculative_tokens
        self.min_match_len = min_match_len
        self.max_entries_per_table = max_entries_per_table
        self.initial_window_size = initial_window_size
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
    
    @classmethod
    def from_speculative_config(cls, spec_config) -> "HSpecConfig":
        """Create HSpecConfig from vLLM speculative config.
        
        Args:
            spec_config: vLLM speculative configuration.
        
        Returns:
            HSpecConfig instance.
        """
        return cls(
            similarity_threshold=getattr(spec_config, 'hspec_similarity_threshold', 0.9),
            num_speculative_tokens=spec_config.num_speculative_tokens,
            min_match_len=getattr(spec_config, 'hspec_min_match_len', 1),
        )
