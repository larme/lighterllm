import torch
import numpy as np
from .base_layer_weight import BaseLayerWeight


class PreAndPostLayerWeight(BaseLayerWeight):
    def __init__(self, data_type, network_config):
        self.data_type_ = data_type
        self.final_layernorm_weight_ = None
        self.wte_weight_ = None
        self.lm_head_weight = None
        self.network_config = network_config

    def load_hf_weights(self, weights):
        # input layernorm params

        vob_size = self.network_config["vocab_size"]
        n_embed = self.network_config["hidden_size"]
        if "model.embed_tokens.weight" in weights:
            # print(weights['model.embed_tokens.weight'].shape)
            self.wte_weight_ = weights['model.embed_tokens.weight'].contiguous().to(self.data_type_).cuda()
        if 'lm_head.weight' in weights:
            # print(weights['lm_head.weight'].shape)
            self.lm_head_weight = weights['lm_head.weight'].contiguous().to(self.data_type_).cuda()
        if 'model.norm.weight' in weights:
            self.final_layernorm_weight_ = weights['model.norm.weight'].to(self.data_type_).cuda()

        return
