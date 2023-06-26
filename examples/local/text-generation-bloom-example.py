# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii

mii_configs = {
    "dtype": "fp16",
    "tensor_parallel": 8,
    "port_number": 50950,
    "meta_tensor": True,
}
name = "microsoft/bloom-deepspeed-inference-fp16"

mii.deploy(task='text-generation',
           model=name,
           deployment_name=name + "_deployment",
           model_path="/data/bloom-mp",
           mii_config=mii_configs)
