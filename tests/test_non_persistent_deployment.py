# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import pytest
import os
import torch
from types import SimpleNamespace
import json
import requests
from .utils import validate_config, dtype, tensor_parallel, load_with_sys_mem, enable_deepspeed, enable_zero, ds_config
import mii


def validate_config(config):
    if (config.model in ['bert-base-uncased']) and (config.mii_config['dtype']
                                                    == 'fp16'):
        pytest.skip(f"Model f{config.model} not supported for FP16")
    elif config.mii_config['dtype'] == "fp32" and "bloom" in config.model:
        pytest.skip('bloom does not support fp32')


''' These fixtures provide default values for the deployment config '''


@pytest.fixture(scope="function", params=[False])
def enable_load_balancing(request):
    return request.param


''' These fixtures provide a local deployment and ensure teardown '''


@pytest.fixture(scope="function")
def mii_configs(
    tmpdir: str,
    dtype: str,
    tensor_parallel: int,
    load_with_sys_mem: bool,
    enable_load_balancing: bool,
):

    # Create a hostfile for DeepSpeed launcher when load_balancing is enabled
    hostfile = os.path.join(tmpdir, "hostfile")
    num_gpu = torch.cuda.device_count()
    enable_load_balancing = enable_load_balancing
    if enable_load_balancing:
        with open(hostfile, "w") as f:
            f.write(f"localhost slots={num_gpu}")

    return {
        'dtype': dtype,
        'tensor_parallel': tensor_parallel,
        'load_with_sys_mem': load_with_sys_mem,
        'enable_load_balancing': enable_load_balancing,
        'replica_num': num_gpu * enable_load_balancing // tensor_parallel,
        'hostfile': hostfile,
    }


@pytest.fixture(scope="function")
def deployment_config(task_name: str,
                      model_name: str,
                      mii_configs: dict,
                      enable_deepspeed: bool,
                      enable_zero: bool,
                      ds_config: dict):
    config = SimpleNamespace(task=task_name,
                             model=model_name,
                             deployment_type=mii.DeploymentType.NON_PERSISTENT,
                             deployment_name=model_name + "_deployment",
                             model_path=os.getenv("TRANSFORMERS_CACHE",
                                                  None),
                             mii_config=mii_configs,
                             enable_deepspeed=enable_deepspeed,
                             enable_zero=enable_zero,
                             ds_config=ds_config)
    validate_config(config)
    return config


@pytest.fixture(scope="function", params=[None])
def expected_failure(request):
    return request.param


@pytest.fixture(scope="function")
def non_persistent_deployment(deployment_config, expected_failure):
    if expected_failure is not None:
        with pytest.raises(expected_failure) as excinfo:
            mii.deploy(**deployment_config.__dict__)
        yield excinfo
    else:
        mii.deploy(**deployment_config.__dict__)
        yield deployment_config
        mii.terminate(deployment_config.deployment_name)


''' Unit tests '''


@pytest.mark.local
@pytest.mark.parametrize("dtype", ['fp16', 'fp32'])
@pytest.mark.parametrize(
    "task_name, model_name, query",
    [
        (
            "conversational",
            "microsoft/DialoGPT-small",
            {
                "text": "DeepSpeed is the greatest",
                "conversation_id": 3,
                "past_user_inputs": [],
                "generated_responses": [],
            },
        ),
        (
            "fill-mask",
            "bert-base-uncased",
            {
                "query": "Hello I'm a [MASK] model."
            },
        ),
        (
            "question-answering",
            "deepset/roberta-large-squad2",
            {
                "question": "What is the greatest?",
                "context": "DeepSpeed is the greatest",
            },
        ),
        (
            "text-generation",
            "distilgpt2",
            {
                "query": ["DeepSpeed is the greatest"]
            },
        ),
        (
            "text-generation",
            "bigscience/bloom-560m",
            {
                "query": ["DeepSpeed is the greatest",
                          'Seattle is']
            },
        ),
        ("token-classification",
         "Jean-Baptiste/roberta-large-ner-english",
         {
             "query": "My name is jean-baptiste and I live in montreal."
         }),
        (
            "text-classification",
            "roberta-large-mnli",
            {
                "query": "DeepSpeed is the greatest"
            },
        ),
    ],
)
def test_single_GPU(non_persistent_deployment, query):
    generator = mii.mii_query_handle(non_persistent_deployment.deployment_name)
    result = generator.query(query)
    assert result


@pytest.mark.local
@pytest.mark.parametrize("enable_load_balancing", [True])
@pytest.mark.parametrize("expected_failure", [AssertionError])
@pytest.mark.parametrize("tensor_parallel", [1, 2])
@pytest.mark.parametrize(
    "task_name, model_name, query",
    [
        (
            "text-generation",
            "bigscience/bloom-560m",
            {
                "query": ["DeepSpeed is the greatest"]
            },
        ),
    ],
)
def test_load_balancing(non_persistent_deployment, query):
    print(f"TESTING NON_PERSISTENT_DEPLOYMENT: {non_persistent_deployment}")
    assert "Cannot use Load Balancing with Non persistent deployment" in str(
        non_persistent_deployment.value)


@pytest.mark.local
@pytest.mark.parametrize("load_with_sys_mem", [True])
@pytest.mark.parametrize(
    "task_name, model_name, query",
    [
        (
            "text-generation",
            "distilgpt2",
            {
                "query": ["DeepSpeed is the greatest"]
            },
        ),
    ],
)
def test_load_to_sys_mem(non_persistent_deployment, query):
    generator = mii.mii_query_handle(non_persistent_deployment.deployment_name)
    result = generator.query(query)
    assert result


def test_zero_config(non_persistent_deployment, query):
    generator = mii.mii_query_handle(non_persistent_deployment.deployment_name)
    result = generator.query(query)
    assert result


@pytest.mark.local
@pytest.mark.parametrize("expected_failure", [AssertionError])
@pytest.mark.parametrize("enable_deepspeed, enable_zero, dtype",
                         [(True,
                           True,
                           'fp32'),
                          (False,
                           True,
                           'fp16')])
@pytest.mark.parametrize("ds_config",
                         [
                             {
                                 "fp16": {
                                     "enabled": False
                                 },
                                 "bf16": {
                                     "enabled": False
                                 },
                                 "zero_optimization": {
                                     "stage": 3,
                                     "offload_param": {
                                         "device": "cpu",
                                     },
                                 },
                                 "train_micro_batch_size_per_gpu": 1,
                             },
                         ])
@pytest.mark.parametrize(
    "task_name, model_name, query",
    [
        (
            "text-generation",
            "distilgpt2",
            {
                "query": "DeepSpeed is the greatest"
            },
        ),
    ],
)
def test_zero_config_fail(non_persistent_deployment, query):
    print(non_persistent_deployment)
    assert "MII Config Error" in str(non_persistent_deployment.value)
