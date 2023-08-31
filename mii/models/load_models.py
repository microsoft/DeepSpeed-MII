# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import mii
import torch
import inspect
import deepspeed
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.zero.config import ZeroStageEnum


def load_models(deployment_config):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    inf_config = {
        "tensor_parallel": {
            "tp_size": deployment_config.tensor_parallel,
            "mpu": None
        },
        "dtype": deployment_config.dtype,
        "replace_method": "auto",
        "enable_cuda_graph": deployment_config.enable_cuda_graph,
        "checkpoint": None,
        "config": None,
        "training_mp_size": 1,
        "replace_with_kernel_inject": deployment_config.replace_with_kernel_inject,
        "max_tokens": deployment_config.max_tokens,
    }

    provider = deployment_config.provider
    if provider == mii.constants.ModelProvider.HUGGING_FACE:
        from mii.models.providers.huggingface import hf_provider

        inference_pipeline = hf_provider(deployment_config)
        if deployment_config.meta_tensor:
            inf_config["checkpoint"] = inference_pipeline.checkpoint_dict
            if deployment_config.dtype == torch.int8:
                # Support for older DeepSpeed versions
                if ("enable_qkv_quantization"
                        in inspect.signature(deepspeed.init_inference).parameters):
                    inf_config["enable_qkv_quantization"] = True
    elif provider == mii.constants.ModelProvider.ELEUTHER_AI:
        assert False, "Eleuther AI support is currently disabled."
        # TODO: Re-enable EleutherAI model support
        """
        from mii.models.providers.eleutherai import eleutherai_provider
        assert mii_config.dtype == torch.half, "gpt-neox only support fp16"
        assert mii_config.enable_cuda_graph == False, "Provider EleutherAI not supported with Cuda Graphs"
        from megatron import mpu
        inf_config["tensor_parallel"]["mpu"] = mpu
        inference_pipeline = eleutherai_provider(model_path,
                                                 model_name,
                                                 task_name,
                                                 mii_config)
        inf_config["training_mp_size"] = 2
        inf_config["config"] = inference_pipeline.neox_args
        """
    elif provider == mii.constants.ModelProvider.DIFFUSERS:
        from mii.models.providers.diffusers import diffusers_provider
        inference_pipeline = diffusers_provider(deployment_config)
    else:
        raise ValueError(f"Unknown model provider {provider}")
    """
    print(
        f"> --------- MII Settings: ds_optimize={ds_optimize}, replace_with_kernel_inject={mii_config.replace_with_kernel_inject}, enable_cuda_graph={mii_config.enable_cuda_graph} "
    )
    """
    if deployment_config.enable_deepspeed:
        engine = deepspeed.init_inference(getattr(inference_pipeline,
                                                  "model",
                                                  inference_pipeline),
                                          config=inf_config)
        if deployment_config.profile_model_time:
            engine.profile_model_time()
        if hasattr(inference_pipeline, "model"):
            inference_pipeline.model = engine

    elif deployment_config.enable_zero:
        ds_config = DeepSpeedConfig(deployment_config.ds_config)
        assert (
            ds_config.zero_optimization_stage == ZeroStageEnum.weights
        ), "DeepSpeed ZeRO inference is only supported for ZeRO-3"

        # initialise Deepspeed ZeRO and store only the engine object
        ds_engine = deepspeed.initialize(model=inference_pipeline.model,
                                         config=deployment_config.ds_config)[0]
        ds_engine.module.eval()  # inference
        inference_pipeline.model = ds_engine.module

    if deployment_config.load_with_sys_mem:
        inference_pipeline.device = torch.device(f"cuda:{local_rank}")

    # Free up memory used when initially loading models
    # so nvidia-smi reports correct amount of memory used.
    torch.cuda.empty_cache()

    return inference_pipeline
