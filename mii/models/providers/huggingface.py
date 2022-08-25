import os
import torch
from transformers import pipeline


def hf_provider(model_path, model_name, task_name, mii_config):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    inference_pipeline = pipeline(task_name,
                                  model=model_name,
                                  device=local_rank,
                                  framework="pt")
    if mii_config.torch_dtype() == torch.half:
        inference_pipeline.model.half()
    return inference_pipeline
