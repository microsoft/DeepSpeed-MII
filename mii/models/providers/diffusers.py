import os
import torch


def diffusers_provider(model_path, model_name, task_name, mii_config):
    from diffusers import DiffusionPipeline
    local_rank = int(os.getenv('LOCAL_RANK', '0'))

    kwargs = {}
    if mii_config.torch_dtype() == torch.half:
        kwargs["torch_dtype"] = torch.float16
        kwargs["revision"] = "fp16"

    pipeline = DiffusionPipeline.from_pretrained(model_name,
                                                 use_auth_token=mii_config.hf_auth_token,
                                                 **kwargs)
    pipeline = pipeline.to(f"cuda:{local_rank}")
    return pipeline
