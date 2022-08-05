'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import mii.policies

supported_models = [
    mii.policies.HFBertLayerPolicy,
    mii.policies.HFGPTNEOLayerPolicy,
    mii.policies.GPTNEOXLayerPolicy,
    mii.policies.HFGPTJLayerPolicy,
    mii.policies.MegatronLayerPolicy,
    mii.policies.HFGPT2LayerPolicy,
    mii.policies.BLOOMLayerPolicy
]
