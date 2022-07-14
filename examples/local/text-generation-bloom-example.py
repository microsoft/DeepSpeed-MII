import os
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface/transformers/'

import mii

mii_configs = {"dtype": "fp16", "tensor_parallel":8}

name = "bigscience/bloom"

mii.deploy('text-generation',
           name,
           mii.DeploymentType.LOCAL,
           deployment_name=name + "_deployment",
           local_model_path="/tmp/huggingface/transformers/",
           mii_configs=mii_configs,
           enable_deepspeed=True,
           enable_zero=False)
