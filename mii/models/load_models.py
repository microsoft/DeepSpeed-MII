'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import mii
from pathlib import Path
import torch
import deepspeed
from deepspeed.runtime.zero.constants import *


def check_zero_ds_config(config):
    config_zero = config.get(ZERO_OPTIMIZATION, {})
    stage = config_zero.get(ZERO_OPTIMIZATION_STAGE, None)
    if stage != ZERO_OPTIMIZATION_WEIGHTS:
        assert False, "DeepSpeed ZeRO inference is only supported for ZeRO 3 optimization stage"


def hf_provider(model_path, model_name, task_name, mii_config):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
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


'''
TODO: The following class and functions are non-optimal (i.e., hacky) solutions
to getting the Bloom models working and will be refactored in a future PR
'''


class BloomPipeline(object):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, inputs, min_length=20, do_sample=False, **kwargs):
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        from deepspeed.inference.engine import InferenceEngine
        if isinstance(self.model, InferenceEngine):
            self.model = self.model.module

        # expand proto list into py-list
        inputs = [i for i in inputs]
        tokens = self.tokenizer.batch_encode_plus(inputs,
                                                  return_tensors="pt",
                                                  padding=True)
        for t in tokens:
            if torch.is_tensor(tokens[t]):
                tokens[t] = tokens[t].to(f'cuda:{local_rank}')
        greedy_output = self.model.generate(**tokens,
                                            min_length=min_length,
                                            max_length=min_length,
                                            do_sample=do_sample)
        outputs = self.tokenizer.batch_decode(greedy_output, skip_special_tokens=True)

        # construct output to align w. HF pipeline
        output_dicts = []
        for output in outputs:
            output_dicts.append([{'generated_text': output}])

        return output_dicts


def get_checkpoint_files(pretrained_model_name_or_path):
    from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME, cached_path, hf_bucket_url, is_offline_mode
    from transformers.utils.hub import EntryNotFoundError
    from transformers.modeling_utils import get_checkpoint_shard_files

    cache_dir = None
    is_sharded = False
    revision = None
    local_files_only = False

    filename = WEIGHTS_NAME
    archive_file = hf_bucket_url(pretrained_model_name_or_path,
                                 filename=filename,
                                 revision=revision)

    try:
        resolved_archive_file = cached_path(
            archive_file,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return [resolved_archive_file]

    except (EntryNotFoundError, FileNotFoundError):
        if filename == WEIGHTS_NAME:
            # Maybe the checkpoint is sharded, we try to grab the index name in this case.
            archive_file = hf_bucket_url(
                pretrained_model_name_or_path,
                filename=WEIGHTS_INDEX_NAME,
                revision=revision,
            )
            resolved_archive_file = cached_path(
                archive_file,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            is_sharded = True

    if is_sharded:
        # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            cache_dir=cache_dir,
            revision=revision
        )

        return resolved_archive_file


def write_checkponts_json(model_name):
    import io
    import json
    checkpoints_json = "checkpoints.json"
    with io.open(checkpoints_json, 'w', encoding='utf-8') as f:

        checkpoint_files = get_checkpoint_files(model_name)

        data = {"type": "BLOOM-176B", "checkpoints": checkpoint_files, "version": 1.0}
        json.dump(data, f)


# TODO: This function is a hack for the Bloom models and will be replaced with a LargeModel provider code path
def load_hf_llm(model_path, model_name, task_name, mii_config):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from transformers import pipeline
    import torch.distributed as dist
    from deepspeed import OnDevice

    deepspeed.init_distributed('nccl')
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    with OnDevice(dtype=torch.float16, enabled=True):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model = model.eval()
    if local_rank == 0:
        write_checkponts_json(model_name)
    dist.barrier()
    inference_pipeline = BloomPipeline(model=model, tokenizer=tokenizer)
    return inference_pipeline


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
    mpu = None
    args = None
    training_mp_size = 1
    if provider == mii.constants.ModelProvider.HUGGING_FACE:
        inference_pipeline = hf_provider(model_path, model_name, task_name, mii_config)
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
    elif provider == mii.constants.ModelProvider.HUGGING_FACE_LLM:
        assert mii_config.torch_dtype() == torch.half, "Bloom models only support fp16"
        assert mii_config.enable_cuda_graph == False, "Bloom models do no support Cuda Graphs"
        inference_pipeline = load_hf_llm(model_path, model_name, task_name, mii_config)
        checkpoint = "checkpoints.json"
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
