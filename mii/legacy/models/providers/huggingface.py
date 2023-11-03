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
from huggingface_hub import snapshot_download

from mii.legacy.utils import mii_cache_path, is_aml


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


def _attempt_load(load_fn, model_name, model_path, cache_path, kwargs={}):
    try:
        value = load_fn(model_name, **kwargs)
    except OSError:
        if is_aml():
            print(f"Attempted load but failed, retrying using model_path={model_path}")
            value = load_fn(model_path, **kwargs)
        else:
            print(f"Attempted load but failed, retrying using cache_dir={cache_path}")
            value = load_fn(model_name, cache_dir=cache_path, **kwargs)
    return value


def create_checkpoint_dict(model_name, model_path, checkpoint_dict):
    if checkpoint_dict:
        return checkpoint_dict
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
    if os.path.isfile(os.path.join(model_path, "ds_inference_config.json")):
        with open(os.path.join(model_path, "ds_inference_config.json")) as f:
            data = json.load(f)
        data["base_dir"] = model_path
        return data
    else:
        checkpoint_files = [
            str(entry).split("/")[-1]
            for entry in Path(model_path).rglob("*.[bp][it][n]") if entry.is_file()
        ]
        data = {
            "type": "DS_MODEL",
            "checkpoints": checkpoint_files,
            "version": 1.0,
            "base_dir": model_path,
        }
        return data


def load_with_meta_tensor(model_config):
    deepspeed.init_distributed("nccl")

    cache_path = mii_cache_path()

    tokenizer = _attempt_load(
        AutoTokenizer.from_pretrained,
        model_config.model,
        model_config.model_path,
        cache_path,
        kwargs={"padding_side": "left"},
    )
    tokenizer.pad_token = tokenizer.eos_token

    config = _attempt_load(AutoConfig.from_pretrained,
                           model_config.model,
                           model_config.model_path,
                           cache_path)

    with OnDevice(dtype=torch.float16, device="meta", enabled=True):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model = model.eval()
    checkpoint_dict = create_checkpoint_dict(model_config.model,
                                             model_config.model_path,
                                             model_config.checkpoint_dict)
    torch.distributed.barrier()
    inference_pipeline = MetaTensorPipeline(model=model,
                                            tokenizer=tokenizer,
                                            checkpoint_dict=checkpoint_dict)
    return inference_pipeline


def hf_provider(model_config):
    if model_config.meta_tensor:
        return load_with_meta_tensor(model_config)
    else:
        device = get_device(load_with_sys_mem=model_config.load_with_sys_mem)
        inference_pipeline = pipeline(
            model_config.task,
            model=model_config.model if not is_aml() else model_config.model_path,
            device=device,
            framework="pt",
            torch_dtype=model_config.dtype,
            **model_config.pipeline_kwargs,
        )
        return inference_pipeline
