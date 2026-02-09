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
HSpec Proposer: Hidden State based Speculative Decoding Proposer.

This module implements the Proposer interface for HSpec,
which uses hidden state similarity matching to propose draft tokens.
"""

from typing import Optional, List

import ray
import torch
import numpy as np
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType
from vllm_ascend.spec_decode.hspec_table import get_hspec_tables, GlobalHSpecTableGroup
from vllm_ascend.spec_decode.hspec_utils import prompt_id_from_token_ids


class HSpecProposer(Proposer):
    """HSpec Proposer that uses hidden state matching for speculative decoding.
    
    This proposer queries the HSpec query tables to find similar hidden states
    from previous epochs and uses the corresponding subsequent tokens as drafts.
    
    Attributes:
        name: The type identifier for this proposer.
        device: The device to run on.
        runner: Reference to the model runner.
        hspec_tables: The global HSpec table manager.
        min_match_len: Minimum sequence length before attempting matching.
        max_draft_tokens: Maximum number of draft tokens to propose.
        similarity_threshold: Threshold for hidden state similarity matching.
    """
    
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner):
        """Initialize the HSpec Proposer.
        
        Args:
            vllm_config: vLLM configuration object.
            device: Device to run computations on.
            runner: Reference to the model runner (for accessing hidden states).
        """
        self.name = SpecDcodeType.HSPEC
        self.device = device
        self.runner = runner
        
        # Get speculative config parameters
        spec_config = vllm_config.speculative_config
        self.max_draft_tokens = spec_config.num_speculative_tokens
        self.similarity_threshold = getattr(spec_config, 'hspec_similarity_threshold', 0.9)
        self.min_match_len = getattr(spec_config, 'hspec_min_match_len', 1)
        
        # Get the global HSpec tables
        self.hspec_tables: GlobalHSpecTableGroup = get_hspec_tables(
            similarity_threshold=self.similarity_threshold
        )
        
        # Cache for accept lengths (used for adaptive window control)
        self._accept_lengths: dict[str, int] = {}
        
        # Hidden state dimension (will be set when first hidden state is received)
        self.hidden_dim: Optional[int] = None
    
    def propose(
        self,
        accept_length: int,
        sampled_token_ids: List[int],
        prompt_token_ids: List[int],
        hidden_state: Optional[np.ndarray] = None
    ) -> List[int]:
        """Propose draft tokens for a single request.
        
        Args:
            accept_length: Number of tokens accepted in the previous iteration.
            sampled_token_ids: List of already sampled token ids.
            prompt_token_ids: Original prompt token ids.
            hidden_state: Current hidden state from the model. Shape: (hidden_dim,)
        
        Returns:
            List of draft token ids, or empty list if no match.
        """
        # Create prompt identifier
        prompt_id = prompt_id_from_token_ids(prompt_token_ids)
        
        # Check if we have enough tokens and hidden state
        if len(sampled_token_ids) < self.min_match_len or hidden_state is None:
            return []
        
        # Query the HSpec tables
        draft_tokens = ray.get(
            self.hspec_tables.query(prompt_id, hidden_state, accept_length)
        )
        
        # Limit to max draft tokens
        if len(draft_tokens) > self.max_draft_tokens:
            draft_tokens = draft_tokens[:self.max_draft_tokens]
        
        return draft_tokens
    
    def propose_batch(
        self,
        accept_length_list: List[int],
        sampled_token_id_list: List[List[int]],
        prompt_token_id_list: List[List[int]],
        hidden_state_list: Optional[List[np.ndarray]] = None
    ) -> List[List[int]]:
        """Propose draft tokens for a batch of requests.
        
        This is the main method called during batch inference.
        
        Args:
            accept_length_list: List of accept lengths for each request.
            sampled_token_id_list: List of sampled token id lists.
            prompt_token_id_list: List of prompt token id lists.
            hidden_state_list: List of hidden states. Each shape: (hidden_dim,)
        
        Returns:
            List of draft token lists for each request.
        """
        batch_size = len(accept_length_list)
        
        # Handle missing hidden states
        if hidden_state_list is None:
            return [[] for _ in range(batch_size)]
        
        # Create prompt identifiers
        prompt_id_list = [
            prompt_id_from_token_ids(prompt_token_ids)
            for prompt_token_ids in prompt_token_id_list
        ]
        
        # Filter requests that have enough tokens
        valid_indices = []
        valid_prompt_ids = []
        valid_hidden_states = []
        valid_accept_lengths = []
        
        for i in range(batch_size):
            if (len(sampled_token_id_list[i]) >= self.min_match_len and 
                hidden_state_list[i] is not None):
                valid_indices.append(i)
                valid_prompt_ids.append(prompt_id_list[i])
                valid_hidden_states.append(hidden_state_list[i])
                valid_accept_lengths.append(accept_length_list[i])
        
        # Initialize results with empty lists
        results = [[] for _ in range(batch_size)]
        
        if not valid_indices:
            return results
        
        # Batch query using ZMQ server for speed
        batch_draft_tokens = self.hspec_tables.post_query_batch(
            valid_prompt_ids,
            valid_hidden_states,
            valid_accept_lengths
        )
        
        # Map results back to original indices
        for idx, draft_tokens in zip(valid_indices, batch_draft_tokens):
            if len(draft_tokens) > self.max_draft_tokens:
                draft_tokens = draft_tokens[:self.max_draft_tokens]
            results[idx] = draft_tokens
        
        return results
    
    def propose_with_hidden_states(
        self,
        valid_sampled_token_ids: List[List[int]],
        hidden_states: torch.Tensor,
        scheduler_output: SchedulerOutput,
        spec_decode_metadata: Optional[SpecDecodeMetadata] = None
    ) -> List[List[int]]:
        """Propose draft tokens using hidden states from model execution.
        
        This method is called from generate_token_ids() after the target model
        has been executed and hidden states are available.
        
        Args:
            valid_sampled_token_ids: List of valid sampled token id lists.
            hidden_states: Hidden states tensor from model. Shape: (batch_size, hidden_dim)
            scheduler_output: Scheduler output containing request information.
            spec_decode_metadata: Spec decode metadata (if available).
        
        Returns:
            List of draft token lists for each request.
        """
        batch_size = len(valid_sampled_token_ids)
        
        if batch_size == 0 or hidden_states is None:
            return []
        
        # Convert hidden states to numpy for the query
        # Take the last position's hidden state for each request
        hidden_states_np = hidden_states.cpu().numpy()
        
        # Set hidden dimension if not set
        if self.hidden_dim is None:
            self.hidden_dim = hidden_states_np.shape[-1]
        
        # Get prompt token ids from scheduler output
        prompt_token_id_list = []
        accept_length_list = []
        
        # TODO: Extract prompt_token_ids and accept_lengths from scheduler_output
        # This requires access to the request states
        for i, req_id in enumerate(scheduler_output.scheduled_new_reqs):
            # Placeholder: In actual implementation, get from request state
            prompt_token_id_list.append([])  # TODO: Get actual prompt token ids
            accept_length_list.append(1)  # TODO: Get actual accept length
        
        # For now, use a simple version
        hidden_state_list = [hidden_states_np[i] for i in range(batch_size)]
        
        return self.propose_batch(
            accept_length_list,
            valid_sampled_token_ids,
            prompt_token_id_list,
            hidden_state_list
        )
    
    def load_model(self, model):
        """Load model weights (not needed for HSpec as it doesn't have a separate model).
        
        Args:
            model: The target model (not used).
        """
        # HSpec doesn't need a separate model, it uses the target model's hidden states
        pass
    
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        skip_attn: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None
    ):
        """Perform a dummy run for warmup (not needed for HSpec).
        
        Args:
            num_tokens: Number of tokens.
            with_prefill: Whether this is a prefill run.
            skip_attn: Whether to skip attention.
            num_reqs: Number of requests.
            num_tokens_across_dp: Token counts across data parallel groups.
            aclgraph_runtime_mode: ACL graph runtime mode.
            batch_descriptor: Batch descriptor.
        """
        # HSpec doesn't need warmup as it doesn't have a neural network component
        pass
    
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
        aux_hidden_states: torch.Tensor = None
    ) -> List[List[int]]:
        """Generate draft token ids using hidden states.
        
        This is the main interface called by model_runner after target model execution.
        
        Args:
            valid_sampled_token_ids: List of valid sampled token id lists.
            sampling_metadata: Sampling metadata.
            scheduler_output: Scheduler output.
            spec_decode_metadata: Spec decode metadata.
            positions: Position tensor.
            num_scheduled_tokens: Number of scheduled tokens.
            hidden_states: Hidden states from target model.
            attn_metadata: Attention metadata.
            aux_hidden_states: Auxiliary hidden states.
        
        Returns:
            List of draft token lists for each request.
        """
        if hidden_states is None:
            # Fallback: return empty draft tokens
            return [[] for _ in range(len(valid_sampled_token_ids))]
        
        # Get the last hidden state for each sequence
        # hidden_states shape: (total_tokens, hidden_dim)
        # We need to extract the last token's hidden state for each request
        
        batch_size = len(valid_sampled_token_ids)
        if batch_size == 0:
            return []
        
        # Extract hidden states at logits positions.
        # NOTE(Ascend): torch.bfloat16 -> numpy is not supported directly.
        # Cast to float16/float32 before transferring to CPU.
        # hidden_states_np = hidden_states.cpu().numpy()
        hs = hidden_states
        if isinstance(hs, torch.Tensor) and hs.dtype == torch.bfloat16:
            hs = hs.to(dtype=torch.float16)
        hidden_states_np = hs.cpu().numpy()
        
        # Build hidden state list for each request
        hidden_state_list = []
        prompt_token_id_list = []
        accept_length_list = []
        
        # Get request information from runner's input batch
        input_batch = self.runner.input_batch
        
        for i, req_id in enumerate(input_batch.req_ids[:batch_size]):
            # Get the hidden state for this request
            # TODO: Properly index hidden states based on token positions
            if i < len(hidden_states_np):
                hidden_state_list.append(hidden_states_np[i])
            else:
                hidden_state_list.append(None)
            
            # Get prompt token ids
            # TODO: Get from request state
            prompt_token_ids = input_batch.token_ids_cpu[i].tolist()
            prompt_token_id_list.append(prompt_token_ids)
            
            # Get accept length from previous iteration
            accept_length = self._accept_lengths.get(req_id, 1)
            accept_length_list.append(accept_length)
        
        # Propose draft tokens
        draft_tokens_list = self.propose_batch(
            accept_length_list,
            valid_sampled_token_ids,
            prompt_token_id_list,
            hidden_state_list
        )
        
        return draft_tokens_list
    
    def update_accept_lengths(self, req_ids: List[str], accept_lengths: List[int]):
        """Update the accept lengths for requests after verification.
        
        Args:
            req_ids: List of request ids.
            accept_lengths: List of accept lengths for each request.
        """
        for req_id, accept_length in zip(req_ids, accept_lengths):
            self._accept_lengths[req_id] = accept_length
    
    def clear_request(self, req_id: str):
        """Clear cached data for a completed request.
        
        Args:
            req_id: The request id to clear.
        """
        if req_id in self._accept_lengths:
            del self._accept_lengths[req_id]
