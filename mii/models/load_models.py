'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import mii
import torch
import deepspeed
from deepspeed.runtime.zero.constants import *


def check_zero_ds_config(config):
    config_zero = config.get(ZERO_OPTIMIZATION, {})
    stage = config_zero.get(ZERO_OPTIMIZATION_STAGE, None)
    if stage != ZERO_OPTIMIZATION_WEIGHTS:
        assert False, "DeepSpeed ZeRO inference is only supported for ZeRO 3 optimization stage"


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

    checkpoint = None
    checkpoint_path = None
    mpu = None
    args = None
    training_mp_size = 1
    if provider == mii.constants.ModelProvider.HUGGING_FACE:
        from mii.models.providers.huggingface import hf_provider
        inference_pipeline = hf_provider(model_path, model_name, task_name, mii_config)
    elif provider == mii.constants.ModelProvider.ELEUTHER_AI:
        from mii.models.providers.eleutherai import eleutherai_provider
        assert mii_config.torch_dtype() == torch.half, "gpt-neox only support fp16"
        assert mii_config.enable_cuda_graph == False, "Provider EleutherAI not supported with Cuda Graphs"
        from megatron import mpu
        inference_pipeline = eleutherai_provider(model_path,
                                                 model_name,
                                                 task_name,
                                                 mii_config)
        training_mp_size = 2
        args = inference_pipeline.neox_args
    elif provider == mii.constants.ModelProvider.HUGGING_FACE_LLM:
        from mii.models.providers.llm import load_hf_llm, _bloom_ckpt_json
        assert mii_config.torch_dtype() == torch.half, "Bloom models only support fp16"
        assert mii_config.enable_cuda_graph == False, "Bloom models do no support Cuda Graphs"
        inference_pipeline = load_hf_llm(model_path, model_name, task_name, mii_config)
        checkpoint = _bloom_ckpt_json()
    else:
        raise ValueError(f"Unknown model provider {provider}")

    if ds_optimize:
        inference_pipeline.model = deepspeed.init_inference(
            inference_pipeline.model,
            mp_size=world_size,
            training_mp_size=training_mp_size,
            mpu=mpu,
            checkpoint=checkpoint,
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
