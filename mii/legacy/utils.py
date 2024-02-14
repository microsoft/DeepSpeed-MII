# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import pickle
import time
import importlib
import torch
import mii.legacy as mii
from types import SimpleNamespace
from huggingface_hub import HfApi

from mii.legacy.models.score.generate import generated_score_path
from mii.legacy.constants import (
    MII_CACHE_PATH,
    MII_CACHE_PATH_DEFAULT,
    ModelProvider,
    SUPPORTED_MODEL_TYPES,
    REQUIRED_KEYS_PER_TASK,
    MII_HF_CACHE_EXPIRATION,
    MII_HF_CACHE_EXPIRATION_DEFAULT,
)

from mii.legacy.config import TaskType


def _get_hf_models_by_type(model_type=None, task=None):
    cache_file_path = os.path.join(mii_cache_path(), "HF_model_cache.pkl")
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
            SimpleNamespace(modelId=m.modelId,
                            pipeline_tag=m.pipeline_tag,
                            tags=m.tags) for m in api.list_models()
        ]
        model_data["cache_time"] = current_time

        # Save the updated cache
        with open(cache_file_path, 'wb') as f:
            pickle.dump(model_data, f)

    # Filter the model list
    models = model_data["model_list"]
    if model_type is not None:
        models = [m for m in models if model_type in m.tags]
    if task is not None:
        models = [m for m in models if m.pipeline_tag == task]

    # Extract model IDs
    model_ids = [m.modelId for m in models]

    if task == TaskType.TEXT_GENERATION:
        # TODO: this is a temp solution to get around some HF models not having the correct tags
        model_ids.extend([
            "microsoft/bloom-deepspeed-inference-fp16",
            "microsoft/bloom-deepspeed-inference-int8",
            "EleutherAI/gpt-neox-20b"
        ])

    return model_ids


def get_supported_models(task):
    supported_models = []

    for model_type, provider in SUPPORTED_MODEL_TYPES.items():
        if provider == ModelProvider.HUGGING_FACE:
            models = _get_hf_models_by_type(model_type, task)
        elif provider == ModelProvider.ELEUTHER_AI:
            if task == TaskType.TEXT_GENERATION:
                models = [model_type]
        elif provider == ModelProvider.DIFFUSERS:
            models = _get_hf_models_by_type(model_type, task)
        supported_models.extend(models)
    if not supported_models:
        raise ValueError(f"Task {task} not supported")

    return supported_models


def check_if_task_and_model_is_supported(task, model_name):
    supported_models = get_supported_models(task)
    assert (
        model_name in supported_models
    ), f"{task} is not supported by {model_name}. This task is supported by {len(supported_models)} other models. See which models with `mii.get_supported_models(mii.{task})`."


def check_if_task_and_model_is_valid(task, model_name):
    valid_task_models = _get_hf_models_by_type(None, task)
    assert (
        model_name in valid_task_models
    ), f"{task} is not supported by {model_name}. This task is supported by {len(valid_task_models)} other models. See which models with `mii.get_supported_models(mii.{task})`."


def full_model_path(model_path):
    aml_model_dir = os.environ.get('AZUREML_MODEL_DIR', None)
    if aml_model_dir:
        # (potentially) append relative model_path w. aml path
        assert os.path.isabs(aml_model_dir), f"AZUREML_MODEL_DIR={aml_model_dir} must be an absolute path"
        if model_path:
            assert not os.path.isabs(model_path), f"model_path={model_path} must be relative to append w. AML path"
            return os.path.join(aml_model_dir, model_path)
        else:
            return aml_model_dir
    elif model_path:
        return model_path
    else:
        return mii.constants.MII_MODEL_PATH_DEFAULT


def is_aml():
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
}


def kwarg_dict_to_proto(kwarg_dict):
    def get_proto_value(value):
        proto_value = mii.grpc_related.proto.legacymodelresponse_pb2.Value()
        setattr(proto_value, dtype_proto_field[type(value)], value)
        return proto_value

    return {k: get_proto_value(v) for k, v in kwarg_dict.items()}


def unpack_proto_query_kwargs(query_kwargs):
    query_kwargs = {
        k: getattr(v,
                   v.WhichOneof("oneof_values"))
        for k,
        v in query_kwargs.items()
    }
    return query_kwargs


def extract_query_dict(task, request_dict):
    required_keys = REQUIRED_KEYS_PER_TASK[task]
    query_dict = {}
    for key in required_keys:
        value = request_dict.pop(key, None)
        if value is None:
            raise ValueError("Request for task: {task} is missing required key: {key}.")
        query_dict[key] = value
    return query_dict


def get_num_gpus(mii_config):
    num_gpus = mii_config.model_config.tensor_parallel

    assert (
        torch.cuda.device_count() >= num_gpus
    ), f"Available GPU count: {torch.cuda.device_count()} does not meet the required gpu count: {num_gpus}"
    return num_gpus


def get_provider(model_name, task):
    if model_name == "gpt-neox":
        provider = ModelProvider.ELEUTHER_AI
    elif task in [TaskType.TEXT2IMG, TaskType.INPAINTING]:
        provider = ModelProvider.DIFFUSERS
    else:
        provider = ModelProvider.HUGGING_FACE
    return provider
