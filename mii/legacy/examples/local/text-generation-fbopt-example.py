# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii

mii_config = {'dtype': 'fp16'}

name = "facebook/opt-1.3b"

ds_config = {
    "fp16": {
        "enabled": True
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
        },
    },
    "train_micro_batch_size_per_gpu": 1,
}

mii.deploy(task='text-generation',
           model=name,
           deployment_name=name + "_deployment",
           model_path=".cache/models/" + name,
           mii_config=mii_config,
           enable_deepspeed=False,
           enable_zero=True,
           ds_config=ds_config)
