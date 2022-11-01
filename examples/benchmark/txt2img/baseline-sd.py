import os
import torch
import diffusers
from utils import benchmark

trials = 5
batch_size = 1
save_path = "."
torch.cuda.manual_seed(42)

# Setup the stable diffusion pipeline via the diffusers pipeline api
pipe = diffusers.StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token=os.environ["HF_AUTH_TOKEN"],
    torch_dtype=torch.float16,
    revision="fp16").to("cuda")

# Create batch size number of prompts
prompts = ["a photo of an astronaut riding a horse on mars"] * batch_size

# Example usage of diffusers pipeline
results = pipe(prompts)
for idx, img in enumerate(results.images):
    img.save(os.path.join(save_path, f"baseline-img{idx}.png"))

# Evaluate performance of pipeline
benchmark(pipe, prompts, save_path, trials, "baseline")
