# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import importlib
import pickle
import os
import time
import deepspeed

from dataclasses import dataclass
from typing import List, TYPE_CHECKING
from datetime import timedelta
from huggingface_hub import HfApi
from transformers import AutoConfig

import mii
from mii.constants import (
    MII_CACHE_PATH,
    MII_CACHE_PATH_DEFAULT,
    REQUIRED_KEYS_PER_TASK,
    MII_HF_CACHE_EXPIRATION,
    MII_HF_CACHE_EXPIRATION_DEFAULT,
)
from mii.logging import logger
from mii.score.generate import generated_score_path

if TYPE_CHECKING:
    from mii.config import ModelConfig


@dataclass
class ModelInfo:
    modelId: str
    pipeline_tag: str
    tags: List[str]


def _hf_model_list() -> List[ModelInfo]:
    cache_file_path = os.path.join(mii_cache_path(), "MII_model_cache.pkl")
    cache_expiration_seconds = os.getenv(MII_HF_CACHE_EXPIRATION,
                                         MII_HF_CACHE_EXPIRATION_DEFAULT)

    # Load or initialize the cache
    model_data = {"cache_time": 0, "model_list": []}
    if os.path.isfile(cache_file_path):
        with open(cache_file_path, 'rb') as f:
            model_data = pickle.load(f)

    current_time = time.time()

    # Update the cache if it has expired
    if (model_data["cache_time"] + cache_expiration_seconds) < current_time:
        api = HfApi()
        model_data["model_list"] = [
            ModelInfo(modelId=m.modelId,
                      pipeline_tag=m.pipeline_tag,
                      tags=m.tags) for m in api.list_models()
        ]
        model_data["cache_time"] = current_time

        # Save the updated cache
        with open(cache_file_path, 'wb') as f:
            pickle.dump(model_data, f)

    return model_data["model_list"]


def get_default_task(model_name_or_path: str) -> str:
    model_name = get_model_name(model_name_or_path)
    models = _hf_model_list()
    for m in models:
        if m.modelId == model_name:
            task = m.pipeline_tag
            logger.info(f"Detected default task as '{task}' for model '{model_name}'")
            return task
    else:
        raise ValueError(f"Model {model_name} not found")


def get_model_name(model_name_or_path: str) -> str:
    model_name = None
    if os.path.exists(model_name_or_path):
        try:
            model_name = AutoConfig.from_pretrained(model_name_or_path)._name_or_path
        except:
            model_name = os.path.basename(model_name_or_path)
            logger.warning(
                f"Could not deduce model name from {model_name_or_path}. Trying with {model_name=} instead."
            )
    else:
        model_name = model_name_or_path
    return model_name


def is_aml() -> bool:
    return os.getenv("AZUREML_MODEL_DIR") is not None


def mii_cache_path():
    cache_path = os.environ.get(MII_CACHE_PATH, MII_CACHE_PATH_DEFAULT)
    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)
    return cache_path


def import_score_file(deployment_name, deployment_type):
    score_path = generated_score_path(deployment_name, deployment_type)
    spec = importlib.util.spec_from_file_location("score", score_path)
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    return score


dtype_proto_field = {
    str: "svalue",
    int: "ivalue",
    float: "fvalue",
    bool: "bvalue",
    dict: "mvalue",
}


def kwarg_dict_to_proto(kwarg_dict):
    def get_proto_value(value):
        proto_value = mii.grpc_related.proto.modelresponse_pb2.Value()

        if isinstance(value, dict):
            nested_dict = mii.grpc_related.proto.modelresponse_pb2.Dictionary()
            for k, v in value.items():
                nested_dict.values[k].CopyFrom(get_proto_value(v))
            proto_value.mvalue.CopyFrom(nested_dict)
        else:
            setattr(proto_value, dtype_proto_field[type(value)], value)

        return proto_value

    return {k: get_proto_value(v) for k, v in kwarg_dict.items()}


def unpack_proto_query_kwargs(query_kwargs):
    def extract_proto_value(proto_value):
        field_name = proto_value.WhichOneof("oneof_values")

        if field_name == "mvalue":
            return {
                k: extract_proto_value(v)
                for k,
                v in proto_value.mvalue.values.items()
            }
        else:
            return getattr(proto_value, field_name)

    return {k: extract_proto_value(v) for k, v in query_kwargs.items()}


def extract_query_dict(task, request_dict):
    required_keys = REQUIRED_KEYS_PER_TASK[task]
    query_dict = {}
    for key in required_keys:
        value = request_dict.pop(key, None)
        if value is None:
            raise ValueError("Request for task: {task} is missing required key: {key}.")
        query_dict[key] = value
    return query_dict


def generate_deployment_name(model_name_or_path: str):
    if os.path.exists(model_name_or_path):
        model_name = os.path.basename(model_name_or_path)
    else:
        model_name = model_name_or_path
    return f"{model_name}-mii-deployment"


def init_distributed(model_config: "ModelConfig"):
    # If not running with a distributed launcher (e.g., deepspeed, torch) set some default environment variables
    required_env = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    if not all([e in os.environ for e in required_env]):
        assert model_config.tensor_parallel == 1, "Attempting to run with TP > 1 and not using a distributed launcher like deepspeed or torch.distributed"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(model_config.torch_dist_port)

    deepspeed.init_distributed(dist_backend="nccl", timeout=timedelta(seconds=1e9))
