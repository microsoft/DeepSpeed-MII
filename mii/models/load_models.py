'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import mii
import torch
import deepspeed
from deepspeed.runtime.constants import *
from pathlib import Path


def check_zero_ds_config(config):
    config_zero = config.get(ZERO_OPTIMIZATION, {})
    stage = config_zero.get(ZERO_OPTIMIZATION_STAGE, None)
    if stage != ZERO_OPTIMIZATION_WEIGHTS:
        assert False, "DeepSpeed ZeRO inference is only supported for ZeRO 3 optimization stage"


def hf_provider(model_path, model_name, task_name, mii_config):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    os.environ['TRANSFORMERS_CACHE'] = str(Path(model_path).resolve())
    from transformers import pipeline
    inference_pipeline = pipeline(task_name,
                                  model=model_name,
                                  device=local_rank,
                                  framework="pt")
    if mii_config.torch_dtype() == torch.half:
        inference_pipeline.model.half()
    return inference_pipeline


def eleutherai_provider(model_path, model_name, task_name, mii_config):
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    from megatron.neox_pipeline import NeoXPipeline
    config = {
        "load": model_path,
        "vocab_file": os.path.join(model_path,
                                   "20B_tokenizer.json"),
        "model_parallel_size": world_size
    }
    return NeoXPipeline(config)


def load_models(task_name,
                model_name,
                model_path,
                ds_optimize,
                ds_zero,
                provider,
                mii_config,
                ds_config_path=None):
    global generator
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    #TODO: pass in mii_config to fetch dtype, training_mp_size, and other params

    if provider == mii.constants.ModelProvider.HUGGING_FACE:
        inference_pipeline = hf_provider(model_path, model_name, task_name, mii_config)
        training_mp_size = 1
        mpu = None
        args = None
    elif provider == mii.constants.ModelProvider.ELEUTHER_AI:
        assert mii_config.torch_dtype() == torch.half, "gpt-neox only support fp16"
        assert mii_config.enable_cuda_graph == False, "Provider EleutherAI not supported with Cuda Graphs"
        from megatron import mpu
        from argparse import Namespace
        inference_pipeline = eleutherai_provider(model_path,
                                                 model_name,
                                                 task_name,
                                                 mii_config)
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
            enable_cuda_graph=mii_config.enable_cuda_graph,
            args=args)
    elif ds_zero:
        assert os.path.exists(ds_config_path), '{ds_config_path} does not exist'
        import json
        ds_config = json.load(open(ds_config_path, "r"))
        check_zero_ds_config(ds_config)

        # initialise Deepspeed ZeRO and store only the engine object
        ds_engine = deepspeed.initialize(model=inference_pipeline.model,
                                         config_params=ds_config)[0]
        ds_engine.module.eval()  # inference
        inference_pipeline.model = ds_engine.module

    return inference_pipeline
