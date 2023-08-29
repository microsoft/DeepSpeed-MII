# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import json
import torch
import deepspeed
from deepspeed.inference.engine import InferenceEngine
from deepspeed import OnDevice
from pathlib import Path
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.utils.hub import EntryNotFoundError
from transformers.modeling_utils import get_checkpoint_shard_files
from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME

from mii.utils import mii_cache_path, is_aml

try:
    from transformers.utils import cached_path, hf_bucket_url

    USE_NEW_HF_CACHE = False
except ImportError:
    from huggingface_hub import snapshot_download

    USE_NEW_HF_CACHE = True


class MetaTensorPipeline(object):
    """
    Class for loading HuggingFace models using meta tensors
    """
    def __init__(self, model, tokenizer, checkpoint_dict):
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_dict = checkpoint_dict

    def __call__(self, inputs, **kwargs):
        device = get_device()
        torch.cuda.set_device(device)
        if isinstance(self.model, InferenceEngine):
            self.model = self.model.module

        # expand proto list into py-list
        inputs = [i for i in inputs]
        tokens = self.tokenizer.batch_encode_plus(inputs,
                                                  return_tensors="pt",
                                                  padding=True)
        for t in tokens:
            if torch.is_tensor(tokens[t]):
                tokens[t] = tokens[t].to(device)

        greedy_output = self.model.generate(**tokens, **kwargs)
        outputs = self.tokenizer.batch_decode(greedy_output, skip_special_tokens=True)

        # construct output to align w. HF pipeline
        output_dicts = []
        for output in outputs:
            output_dicts.append([{"generated_text": output}])

        return output_dicts


def get_device(load_with_sys_mem=False):
    if load_with_sys_mem:
        device = torch.device("cpu")
    else:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
    return device


def _attempt_load(load_fn, model_name, cache_path, kwargs={}):
    try:
        value = load_fn(model_name, **kwargs)
    except OSError:
        print(f"Attempted load but failed, retrying using cache_dir={cache_path}")
        value = load_fn(model_name, cache_dir=cache_path, **kwargs)
    return value


def get_checkpoint_files(pretrained_model_name_or_path):
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
            revision=revision,
        )

        return resolved_archive_file


def create_checkpoint_dict(model_name, model_path, checkpoint_dict):
    if USE_NEW_HF_CACHE:
        model_path = snapshot_download(
            model_name,
            cache_dir=model_path,
            allow_patterns=[
                "*.bin",
                "*.json",
                "*.pt",
            ],
            revision=None,
        )
    if checkpoint_dict:
        checkpoint_dict["base_dir"] = model_path
        return checkpoint_dict
    elif os.path.isfile(os.path.join(model_path, "ds_inference_config.json")):
        with open(os.path.join(model_path, "ds_inference_config.json")) as f:
            data = json.load(f)
        data["base_dir"] = model_path
        return data
    else:
        if USE_NEW_HF_CACHE:
            checkpoint_files = [
                str(entry).split("/")[-1]
                for entry in Path(model_path).rglob("*.[bp][it][n]") if entry.is_file()
            ]
        else:
            checkpoint_files = get_checkpoint_files(model_name)
        data = {
            "type": "DS_MODEL",
            "checkpoints": checkpoint_files,
            "version": 1.0,
            "base_dir": model_path,
        }
        return data


def load_with_meta_tensor(deployment_config):
    deepspeed.init_distributed("nccl")

    cache_path = mii_cache_path()

    tokenizer = _attempt_load(
        AutoTokenizer.from_pretrained,
        deployment_config.model,
        cache_path,
        kwargs={"padding_side": "left"},
    )
    tokenizer.pad_token = tokenizer.eos_token

    config = _attempt_load(AutoConfig.from_pretrained,
                           deployment_config.model,
                           cache_path)

    with OnDevice(dtype=torch.float16, device="meta", enabled=True):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model = model.eval()
    checkpoint_dict = create_checkpoint_dict(deployment_config.model,
                                             deployment_config.model_path,
                                             deployment_config.checkpoint_dict)
    torch.distributed.barrier()
    inference_pipeline = MetaTensorPipeline(model=model,
                                            tokenizer=tokenizer,
                                            checkpoint_dict=checkpoint_dict)
    return inference_pipeline


def hf_provider(deployment_config):
    if deployment_config.meta_tensor:
        return load_with_meta_tensor(deployment_config)
    else:
        device = get_device(load_with_sys_mem=deployment_config.load_with_sys_mem)
        inference_pipeline = pipeline(
            deployment_config.task,
            model=deployment_config.model,
            device=device,
            framework="pt",
            token=deployment_config.hf_auth_token,
            torch_dtype=deployment_config.dtype,
            trust_remote_code=deployment_config.trust_remote_code,
        )
        return inference_pipeline
