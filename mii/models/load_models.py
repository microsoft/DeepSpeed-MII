import os
import torch
import deepspeed


def load_generator_models(name, model_path):
    global generator
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))   
    os.environ['TRANSFORMERS_CACHE'] = model_path
    from transformers import pipeline
    generator = pipeline('text-generation', model=name,
                        device=local_rank)

    generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                           replace_with_kernel_inject=True,
                                           replace_method='auto')

    return generator

