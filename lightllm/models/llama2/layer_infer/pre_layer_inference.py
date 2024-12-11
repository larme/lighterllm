import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.llama2.layer_weights.pre_and_post_layer_weight import PreAndPostLayerWeight
from lightllm.models.llama2.layer_infer.infer_struct import InferStateInfo
from lightllm.utils.infer_utils import mark_cost_time

torch.backends.cudnn.enabled = True


class PreLayerInfer:
    """
    """

    def __init__(self, network_config):
        self.network_config_ = network_config
        self.tp_vocab_size_ = network_config["vocab_size"]
        self.embed_dim_ = network_config["hidden_size"]
        self.layer_norm_eps_ = network_config["rms_norm_eps"]
        self.vob_start_id_ = 0
        self.vob_end_id_ = self.tp_vocab_size_

    @mark_cost_time("pre context forward")
    def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: PreAndPostLayerWeight):
        total_token_num = infer_state.total_token_num
        input_ids = input_ids[0:total_token_num]

        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        return input_embdings

    def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: PreAndPostLayerWeight):
        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        return input_embdings
