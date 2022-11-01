import os
import mii
import torch
from utils import benchmark

trials = 5
batch_size = 1
save_path = "."
deploy_name = "sd_deploy"
torch.cuda.manual_seed(42)

# Deploy Stable Diffusion w. MII
mii_config = {"dtype": "fp16", "hf_auth_token": os.environ["HF_AUTH_TOKEN"]}
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
