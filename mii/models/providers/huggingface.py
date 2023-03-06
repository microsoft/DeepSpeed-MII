import os
import torch
from transformers import pipeline


def hf_provider(model_path, model_name, task_name, mii_config):
    if mii_config.load_with_sys_mem:
        device = torch.device("cpu")
    else:
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        device = torch.device(f"cuda:{local_rank}")
    # We must load in fp16 for int8 because HF doesn't support int8
    torch_dtype = torch.float16 if (mii_config.dtype == torch.int8) else mii_config.dtype
    inference_pipeline = pipeline(
        task_name,
        model=model_name,
        device=device,
        framework="pt",
        use_auth_token=mii_config.hf_auth_token,
        torch_dtype=torch_dtype,
    )
    return inference_pipeline
