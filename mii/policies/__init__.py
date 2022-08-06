'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
from .bert import HFBertLayerPolicy
from .gpt_neo import HFGPTNEOLayerPolicy
from .gpt_neox import GPTNEOXLayerPolicy
from .gptj import HFGPTJLayerPolicy
from .megatron import MegatronLayerPolicy
from .gpt2 import HFGPT2LayerPolicy
from .bloom import BLOOMLayerPolicy

replace_policies = [
    HFBertLayerPolicy,
    HFGPTNEOLayerPolicy,
    GPTNEOXLayerPolicy,
    HFGPTJLayerPolicy,
    MegatronLayerPolicy,
    HFGPT2LayerPolicy,
    BLOOMLayerPolicy
]
