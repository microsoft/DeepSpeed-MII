import os
import torch
from transformers import pipeline


def hf_provider(model_path, model_name, task_name, mii_config):
    if mii_config.load_with_sys_mem:
        device = torch.device("cpu")
    else:
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        device = torch.device(f"cuda:{local_rank}")
    inference_pipeline = pipeline(
        task_name,
        model=model_name,
        device=device,
        framework="pt",
        use_auth_token=mii_config.hf_auth_token,
        torch_dtype=mii_config.dtype,
    )
    return inference_pipeline
