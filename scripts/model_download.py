#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import argparse

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer, AutoModel


def dir_path(path_str):
    if os.path.isdir(path_str):
        return path_str
    elif input(f"{path_str} does not exist, create directory? [y/n]").lower() == "y":
        os.makedirs(path_str)
        return path_str
    else:
        raise NotADirectoryError(path_str)


class HFModelNotFoundError(Exception):
    def __init__(self, model_str):
        super().__init__(f"HuggingFace model not found: '{model_str}'")


def hf_model(model_str):
    api = HfApi()
    models = [m.modelId for m in api.list_models()]
    if model_str in models:
        return model_str
    else:
        raise HFModelNotFoundError(model_str)


parser = argparse.ArgumentParser()
parser.add_argument("--model_path",
                    '-o',
                    type=dir_path,
                    required=True,
                    help="Output directory for downloaded model files")
parser.add_argument("--model_name",
                    '-m',
                    type=hf_model,
                    required=True,
                    help="HuggingFace model name")
args = parser.parse_args()

for auto_func in [AutoConfig, AutoTokenizer, AutoModel]:
    auto_func.from_pretrained(args.model_name, cache_dir=args.model_path)

print(f"Cached files for '{args.model_name}' downloaded to '{args.model_path}'")
