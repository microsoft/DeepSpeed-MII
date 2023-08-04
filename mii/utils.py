# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import sys
import os
import logging
import importlib
import torch
import mii
from huggingface_hub import HfApi

from mii.config import Tasks
from mii.constants import SUPPORTED_MODEL_TYPES, ModelProvider, MII_CACHE_PATH, MII_CACHE_PATH_DEFAULT



def _get_hf_models_by_type(model_type, task=None):
    api = HfApi()
    models = api.list_models(filter=model_type)
    models = ([m.modelId for m in models]
              if task is None else [m.modelId for m in models if m.pipeline_tag == task])
    if task == Tasks.TEXT_GENERATION:
        # TODO: this is a temp solution to get around some HF models not having the correct tags
        models.append("microsoft/bloom-deepspeed-inference-fp16")
        models.append("microsoft/bloom-deepspeed-inference-int8")
        models.append("EleutherAI/gpt-neox-20b")
    return models


# TODO read this from a file containing list of files supported for each task
def _get_supported_models_name(task):
    supported_models = []
    task_name = task

    for model_type, provider in SUPPORTED_MODEL_TYPES.items():
        if provider == ModelProvider.HUGGING_FACE:
            models = _get_hf_models_by_type(model_type, task_name)
        elif provider == ModelProvider.ELEUTHER_AI:
            if task_name == Tasks.TEXT_GENERATION:
                models = [model_type]
        elif provider == ModelProvider.DIFFUSERS:
            models = _get_hf_models_by_type(model_type, task_name)
        supported_models.extend(models)
    if not supported_models:
        raise ValueError(f"Task {task} not supported")

    return supported_models


def check_if_task_and_model_is_supported(task, model_name):
    supported_models = _get_supported_models_name(task)
    assert model_name in supported_models, f"{task} only supports {supported_models}"


def check_if_task_and_model_is_valid(task, model_name):
    task_name = task
    valid_task_models = _get_hf_models_by_type(None, task_name)
    assert (
        model_name in valid_task_models
    ), f"{task_name} only supports {valid_task_models}"


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


def import_score_file(deployment_name):
    spec = importlib.util.spec_from_file_location(
        "score",
        os.path.join(mii_cache_path(),
                     deployment_name,
                     "score.py"))
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
        proto_value = mii.grpc_related.proto.modelresponse_pb2.Value()
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


def get_provider_name(model_name, task):
    if model_name == "gpt-neox":
        provider = mii.constants.MODEL_PROVIDER_NAME_EA
    elif task == mii.Tasks.TEXT2IMG:
        provider = mii.constants.MODEL_PROVIDER_NAME_DIFFUSERS
    else:
        provider = mii.constants.MODEL_PROVIDER_NAME_HF
    return provider


log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LoggerFactory:
    @staticmethod
    def create_logger(name=None, level=logging.INFO):
        """create a logger
        Args:
            name (str): name of the logger
            level: level of logger
        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] "
            "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
        return logger_


logger = LoggerFactory.create_logger(name="MII", level=logging.INFO)
