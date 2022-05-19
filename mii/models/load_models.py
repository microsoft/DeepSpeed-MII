'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import mii
import torch
import deepspeed


def hf_provider(model_path, model_name, task_name, mii_config):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    os.environ['TRANSFORMERS_CACHE'] = model_path
    from transformers import pipeline
    pipeline = pipeline(task_name, model=model_name, device=local_rank)
    if mii_config.torch_dtype() == torch.half:
        inference_pipeline.model.half()
    return pipeline


def eleutherai_provider(model_path, model_name, task_name, mii_config):
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    assert mii_config.torch_dtype() == torch.half, "gpt-neox only support fp16"
    from megatron.neox_pipeline import NeoXPipeline
    config = {
        "load": model_path,
        "vocab_file": os.path.join(model_path,
                                   "20B_tokenizer.json"),
        "model_parallel_size": world_size
    }
    return NeoXPipeline(config)


def load_models(task_name, model_name, model_path, ds_optimize, provider, mii_config):
    global generator
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    #TODO: pass in mii_config to fetch dtype, training_mp_size, and other params

    if provider == mii.constants.ModelProvider.HUGGING_FACE:
        inference_pipeline = hf_provider(model_path, model_name, task_name, mii_config)
        training_mp_size = 1
        mpu = None
        args = None
    elif provider == mii.constants.ModelProvider.ELEUTHER_AI:
        from megatron import mpu
        from argparse import Namespace
        inference_pipeline = eleutherai_provider(model_path, model_name, task_name, mii_config)
        training_mp_size = 2
        args = inference_pipeline.neox_args
    else:
        raise ValueError(f"Unknown model provider {provider}")

    if ds_optimize:
        inference_pipeline.model = deepspeed.init_inference(
            inference_pipeline.model,
            mp_size=world_size,
            training_mp_size=training_mp_size,
            mpu=mpu,
            dtype=mii_config.torch_dtype(),
            replace_with_kernel_inject=True,
            replace_method='auto',
            args=args)

    return inference_pipeline
