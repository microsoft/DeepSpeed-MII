# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import mii
from utils import benchmark

# Get HF auth key from environment or replace with key
hf_auth_key = os.environ["HF_AUTH_TOKEN"]

trials = 10
batch_size = 1
save_path = "."
deploy_name = "sd_deploy"

# Deploy Stable Diffusion w. MII
mii_config = {"dtype": "fp16", "hf_auth_token": hf_auth_key}
mii.deploy(task='text-to-image',
           model="CompVis/stable-diffusion-v1-4",
           deployment_name=deploy_name,
           mii_config=mii_config)

# Example usage of MII deployment
pipe = mii.mii_query_handle(deploy_name)
prompts = {"query": ["a photo of an astronaut riding a horse on mars"] * batch_size}
results = pipe.query(prompts)
for idx, img in enumerate(results.images):
    img.save(os.path.join(save_path, f"mii-img{idx}.png"))

# Evaluate performance of MII
benchmark(pipe.query, prompts, save_path, trials, "mii")

# Tear down the persistent deployment
mii.terminate(deploy_name)
