'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import mii
import json
import torch
import inspect
import deepspeed
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.zero.config import ZeroStageEnum


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

    ds_kwargs = {
        "checkpoint": None,
        "mpu": None,
        "args": None,
        "training_mp_size": 1,
        "replace_with_kernel_inject": mii_config.replace_with_kernel_inject
    }

    if provider == mii.constants.ModelProvider.HUGGING_FACE:
        from mii.models.providers.huggingface import hf_provider
        inference_pipeline = hf_provider(model_path, model_name, task_name, mii_config)
    elif provider == mii.constants.ModelProvider.ELEUTHER_AI:
        from mii.models.providers.eleutherai import eleutherai_provider
        assert mii_config.torch_dtype() == torch.half, "gpt-neox only support fp16"
        assert mii_config.enable_cuda_graph == False, "Provider EleutherAI not supported with Cuda Graphs"
        from megatron import mpu
        ds_kwargs["mpu"] = mpu
        inference_pipeline = eleutherai_provider(model_path,
                                                 model_name,
                                                 task_name,
                                                 mii_config)
        ds_kwargs["training_mp_size"] = 2
        ds_kwargs["args"] = inference_pipeline.neox_args
    elif provider == mii.constants.ModelProvider.HUGGING_FACE_LLM:
        from mii.models.providers.llm import load_hf_llm
        assert mii_config.torch_dtype() == torch.half or mii_config.torch_dtype() == torch.int8, "Bloom models only support fp16/int8"
        assert mii_config.enable_cuda_graph == False, "Bloom models do no support Cuda Graphs"
        inference_pipeline = load_hf_llm(model_path, model_name, task_name, mii_config)
        ds_kwargs["checkpoint"] = inference_pipeline.checkpoint_dict
        if mii_config.torch_dtype() == torch.int8:
            if "enable_qkv_quantization" in inspect.signature(
                    deepspeed.init_inference).parameters:
                ds_kwargs["enable_qkv_quantization"] = True
    elif provider == mii.constants.ModelProvider.DIFFUSERS:
        from mii.models.providers.diffusers import diffusers_provider
        assert not mii_config.enable_cuda_graph, "Diffusers models do no support Cuda Graphs (yet)"
        inference_pipeline = diffusers_provider(model_path,
                                                model_name,
                                                task_name,
                                                mii_config)
        ds_kwargs["replace_with_kernel_inject"] = False  #not supported yet
    else:
        raise ValueError(f"Unknown model provider {provider}")

    print(
        f"> --------- MII Settings: {ds_optimize=}, replace_with_kernel_inject={mii_config.replace_with_kernel_inject}, enable_cuda_graph={mii_config.enable_cuda_graph} "
    )
    if ds_optimize:
        engine = deepspeed.init_inference(getattr(inference_pipeline,
                                                  "model",
                                                  inference_pipeline),
                                          mp_size=world_size,
                                          dtype=mii_config.torch_dtype(),
                                          replace_method='auto',
                                          enable_cuda_graph=mii_config.enable_cuda_graph,
                                          **ds_kwargs)
        if mii_config.profile_model_time:
            engine.profile_model_time()
        if hasattr(inference_pipeline, "model"):
            inference_pipeline.model = engine

    elif ds_zero:
        ds_config = DeepSpeedConfig(ds_config_path)
        #TODO: don't read ds-config from disk, we should pass this around as a dict instead
        ds_config_dict = json.load(open(ds_config_path, 'r'))
        assert ds_config.zero_optimization_stage == ZeroStageEnum.weights, "DeepSpeed ZeRO inference is only supported for ZeRO-3"

        # initialise Deepspeed ZeRO and store only the engine object
        ds_engine = deepspeed.initialize(model=inference_pipeline.model,
                                         config_params=ds_config_dict)[0]
        ds_engine.module.eval()  # inference
        inference_pipeline.model = ds_engine.module
    return inference_pipeline
