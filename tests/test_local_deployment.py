# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import pytest
import os
import torch
from types import SimpleNamespace
import json
import requests

import mii


def validate_config(config):
    if (config.model in ['bert-base-uncased']) and (config.mii_config['dtype']
                                                    == 'fp16'):
        pytest.skip(f"Model f{config.model} not supported for FP16")
    elif config.mii_config['dtype'] == "fp32" and "bloom" in config.model:
        pytest.skip('bloom does not support fp32')


''' These fixtures provide default values for the deployment config '''


@pytest.fixture(scope="function", params=['fp16'])
def dtype(request):
    return request.param


@pytest.fixture(scope="function", params=[1])
def tensor_parallel(request):
    return request.param


@pytest.fixture(scope="function", params=[50050])
def port_number(request):
    return request.param


@pytest.fixture(scope="function", params=[False])
def load_with_sys_mem(request):
    return request.param


@pytest.fixture(scope="function", params=[False])
def enable_load_balancing(request):
    return request.param


@pytest.fixture(scope="function", params=[False])
def enable_restful_api(request):
    return request.param


@pytest.fixture(scope="function", params=[0])
def restful_api_port(request):
    return request.param


@pytest.fixture(scope="function", params=[True])
def enable_deepspeed(request):
    return request.param


@pytest.fixture(scope="function", params=[False])
def enable_zero(request):
    return request.param


@pytest.fixture(scope="function", params=[{}])
def ds_config(request):
    return request.param


''' These fixtures provide a local deployment and ensure teardown '''


@pytest.fixture(scope="function")
def mii_configs(
    tmpdir: str,
    dtype: str,
    tensor_parallel: int,
    port_number: int,
    load_with_sys_mem: bool,
    enable_load_balancing: bool,
    enable_restful_api: bool,
    restful_api_port: int,
):

    # Create a hostfile for DeepSpeed launcher when load_balancing is enabled
    hostfile = os.path.join(tmpdir, "hostfile")
    num_gpu = torch.cuda.device_count()
    enable_load_balancing = enable_load_balancing or enable_restful_api
    if enable_load_balancing:
        with open(hostfile, "w") as f:
            f.write(f"localhost slots={num_gpu}")

    return {
        'dtype': dtype,
        'tensor_parallel': tensor_parallel,
        'port_number': port_number,
        'load_with_sys_mem': load_with_sys_mem,
        'enable_load_balancing': enable_load_balancing,
        'replica_num': num_gpu * enable_load_balancing // tensor_parallel,
        'hostfile': hostfile,
        'enable_restful_api': enable_restful_api,
        'restful_api_port': restful_api_port,
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
                             deployment_type=mii.DeploymentType.LOCAL,
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
def local_deployment(deployment_config, expected_failure):
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
def test_single_GPU(local_deployment, query):
    generator = mii.mii_query_handle(local_deployment.deployment_name)
    result = generator.query(query)
    assert result


@pytest.mark.local
@pytest.mark.parametrize("enable_load_balancing", [True])
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
def test_load_balancing(local_deployment, query):
    generator = mii.mii_query_handle(local_deployment.deployment_name)
    for _ in range(10):
        result = generator.query(query)
    assert result


@pytest.mark.local
@pytest.mark.parametrize("enable_restful_api", [True])
@pytest.mark.parametrize("restful_api_port", [28080])
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
def test_restful_api(local_deployment, query, restful_api_port):
    generator = mii.mii_query_handle(local_deployment.deployment_name)
    for _ in range(2):
        result = generator.query(query)

    url = f'http://localhost:{restful_api_port}/mii/{local_deployment.deployment_name}'
    params = {"request": query}
    json_params = json.dumps(params)
    result = requests.post(url,
                           data=json_params,
                           headers={"Content-Type": "application/json"})
    assert result.status_code == 200
    assert "response" in result.json()


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
def test_load_to_sys_mem(local_deployment, query):
    generator = mii.mii_query_handle(local_deployment.deployment_name)
    result = generator.query(query)
    assert result


@pytest.mark.local
@pytest.mark.parametrize(
    "task_name, model_name, query, enable_deepspeed, enable_zero, ds_config",
    [
        (
            "text-generation",
            "distilgpt2",
            {
                "query": "DeepSpeed is the greatest"
            },
            False,
            True,
            {
                "fp16": {
                    "enabled": True
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
        ),
    ],
)
def test_zero_config(local_deployment, query):
    generator = mii.mii_query_handle(local_deployment.deployment_name)
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
def test_zero_config_fail(local_deployment, query):
    print(local_deployment)
    assert "MII Config Error" in str(local_deployment.value)


''' Not working yet
@pytest.mark.local
@pytest.mark.parametrize(
        "task_name, model_name, config, query",
        [
            (
            "text-generation",
            "gpt-neox",
            {"tensor_parallel": 2, 'dtype': 'fp16'},
            {"query": "DeepSpeed is the greatest"},
            )
        ]
    )
def test_multi_GPU_local_deployment(task_name:str, model_name:str, config:dict, query:dict):
    result = deploy_query_local(task_name=task_name, model_name=model_name, config=config, query=query)
    assert result
'''
