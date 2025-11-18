from typing import Optional

import ray
import enum
import torch
import numpy as np
from typing import Optional
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType
from vllm_ascend.spec_decode.global_module.prefix_tree import get_history_trees

class HistoryRolloutProposer(Proposer):
    def __init__(self, vllm_config, device, runner):
        self.name = SpecDcodeType.HISTO
        self.device = device
        self.runner = runner
        
        # Minimum length of the HistoryRolloutTree to match.
        self.min_n = 2
        # Maximum length of the HistoryRolloutTree to match.
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # self.k = vllm_config.speculative_config.num_speculative_tokens
        self.prompt_lookup = vllm_config.speculative_config.prompt_lookup_max
        self.history_trees = get_history_trees()

    def propose(
        self,
        accept_length: int,
        sampled_token_ids: list[int],
        prompt_token_ids: list[int]
    ) -> Optional[np.ndarray]:
        """Proposes the next sequence of tokens based on history rollout
        speculative decoding pattern.
        """
        prompt_id = str(hash(tuple(prompt_token_ids)))
        
        draft_tokens = []
        if len(sampled_token_ids) >= self.min_n:
            prefix = sampled_token_ids[-self.min_n:]
            draft_tokens = ray.get(self.history_trees.predict(prompt_id, prefix, accept_length))
        return draft_tokens

    def propose_batch(
        self,
        accept_length_list: list[int],
        sampled_token_id_list: list[list[int]],
        prompt_token_id_list: list[list[int]]
    ) -> Optional[np.ndarray]:
        """Proposes the next sequence of tokens based on history rollout
        speculative decoding pattern.
        """
        batch_size = len(accept_length_list)
        prompt_id_list = [str(hash(tuple(prompt_token_id))) for prompt_token_id in prompt_token_id_list]
        prefix_length_list = [self.min_n for _ in range(batch_size)]
        batch_draft_tokens = self.history_trees.post_predict_batch(
            prompt_id_list,
            sampled_token_id_list,
            prefix_length_list,
            accept_length_list
        )
        return batch_draft_tokens

    def load_model(self, model):
        # No model to load.
        pass

    def dummy_run(self,
                  num_tokens: int,
                  with_prefill: bool = False,
                  skip_attn: bool = False,
                  num_reqs: int = 0,
                  num_tokens_across_dp: Optional[torch.Tensor] = None,
                  aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
                  batch_descriptor=None):
        pass

    def generate_token_ids(self,
                           valid_sampled_token_ids: list[list[int]],
                           sampling_metadata: SamplingMetadata = None,
                           scheduler_output: SchedulerOutput = None,
                           spec_decode_metadata: SpecDecodeMetadata = None,
                           positions: torch.Tensor = None,
                           num_scheduled_tokens: int = 0,
                           hidden_states: torch.Tensor = None,
                           attn_metadata=None,
                           aux_hidden_states: torch.Tensor = None):
        pass