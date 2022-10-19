'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import torch
from torch.nn.parameter import Parameter
from deepspeed.module_inject.base_policy import InjectBasePolicy


class HFGPTJLayerPolicy(InjectBasePolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True):
        super().__init__(inference, scale_attention=True)
        self.client_module = client_module
        try:
            import transformers
            HFGPTJLayerPolicy._orig_layer_class = transformers.models.gptj.modeling_gptj.GPTJBlock
        except:
            HFGPTJLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attn.q_proj.weight.shape[1], \
                self.client_module.attn.num_attention_heads

    def attention(self):
        qw = self.client_module.attn.q_proj.weight
        kw = self.client_module.attn.k_proj.weight
        vw = self.client_module.attn.v_proj.weight

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)

        return self.linear_layer, \
                qkvw, \
                None, \
                self.client_module.attn.out_proj.weight, \
                None, \
                self.scale_attention, \
               self.is_megatron_v2

    def mlp(self):
        return self.linear_layer, \
                self.client_module.mlp.fc_in.weight, \
                self.client_module.mlp.fc_in.bias, \
                self.client_module.mlp.fc_out.weight, \
                self.client_module.mlp.fc_out.bias

    def layerNorm(self):
        return None, \
               None, \
               self.client_module.ln_1.weight, \
               self.client_module.ln_1.bias
