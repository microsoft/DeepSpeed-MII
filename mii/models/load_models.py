'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import mii
import torch
import deepspeed

from transformers import AutoConfig

def hf_provider(model_path, model_name, task_name, mii_config):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    os.environ['TRANSFORMERS_CACHE'] = model_path
    from transformers import pipeline
    inference_pipeline = pipeline(task_name, model=model_name, device=local_rank)
    if mii_config.torch_dtype() == torch.half:
        inference_pipeline.model.half()
    return inference_pipeline


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


def load_models(task_name, model_name, model_path, ds_optimize, ds_zero, provider, mii_config):
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
            args=args)
    elif ds_zero:
        config = AutoConfig.from_pretrained(model_name)
        model_hidden_size = config.n_embd

        ds_config = {
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": False
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "nvme",
                    "nvme_path": "/mnt/nvme0/offload",
                    "pin_memory": True,
                    "buffer_count": 6,
                    "buffer_size": 1e9,
                    "max_in_cpu": 1e9
                },
                "aio": {
                    "block_size": 262144,
                    "queue_depth": 32,
                    "thread_count": 1,
                    "single_submit": False,
                    "overlap_events": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": model_hidden_size * model_hidden_size,
                "stage3_prefetch_bucket_size":
                0.1 * model_hidden_size * model_hidden_size,
                "stage3_max_live_parameters": 1e8,
                "stage3_max_reuse_distance": 1e8,
                "stage3_param_persistence_threshold": 10 * model_hidden_size
            },
            "steps_per_print": 2000,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False
        }

        # initialise Deepspeed ZeRO and store only the engine object
        ds_engine = deepspeed.initialize(model=inference_pipeline.model,
                                         config_params=ds_config)[0]
        ds_engine.module.eval()  # inference
        inference_pipeline.model = ds_engine.module

    return inference_pipeline
