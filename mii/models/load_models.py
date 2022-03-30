'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import torch
import deepspeed


def load_models(task_name, model_name, model_path):
    global generator
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    os.environ['TRANSFORMERS_CACHE'] = model_path
    from transformers import pipeline
    inference_pipeline = pipeline(task_name, model=model_name, device=local_rank)

    inference_pipeline.model = deepspeed.init_inference(inference_pipeline.model,
                                                        mp_size=world_size,
                                                        dtype=torch.float,
                                                        replace_with_kernel_inject=True,
                                                        replace_method='auto')

    return inference_pipeline
