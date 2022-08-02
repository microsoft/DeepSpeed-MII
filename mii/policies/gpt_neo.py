'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import torch
from torch.nn.parameter import Parameter
from deepspeed.module_inject.base_policy import InjectBasePolicy


class HFGPTNEOLayerPolicy(InjectBasePolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True):
        super().__init__(inference, scale_attention=False)
        self.client_module = client_module
        try:
            import transformers
            HFGPTNEOLayerPolicy._orig_layer_class = transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoBlock
        except:
            HFGPTNEOLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attn.attention.q_proj.weight.shape[1], \
                self.client_module.attn.attention.num_heads

    def attention(self):
        qw = self.client_module.attn.attention.q_proj.weight
        kw = self.client_module.attn.attention.k_proj.weight
        vw = self.client_module.attn.attention.v_proj.weight

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)

        return self.linear_layer, \
                qkvw, \
                None, \
                self.client_module.attn.attention.out_proj.weight, \
                self.client_module.attn.attention.out_proj.bias, \
                self.scale_attention, \
               self.is_megatron_v2

    def mlp(self):
        return self.linear_layer, \
                self.client_module.mlp.c_fc.weight, \
                self.client_module.mlp.c_fc.bias, \
                self.client_module.mlp.c_proj.weight, \
                self.client_module.mlp.c_proj.bias

    def layerNorm(self):
        return self.client_module.ln_2.weight, \
               self.client_module.ln_2.bias, \
               self.client_module.ln_1.weight, \
               self.client_module.ln_1.bias
