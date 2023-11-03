# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import json
import requests
import mii.legacy as mii
from mii.legacy import pydantic_v1


@pytest.mark.deepspeed
@pytest.mark.parametrize("meta_tensor", [True])
@pytest.mark.parametrize("tensor_parallel", [2])
def test_meta_tensor(deployment, query):
    generator = mii.mii_query_handle(deployment)
    result = generator.query(query)
    assert result


@pytest.mark.parametrize("enable_restful_api", [True])
def test_restful_api(deployment, query, restful_api_port):
    generator = mii.mii_query_handle(deployment)
    for _ in range(2):
        result = generator.query(query)

    url = f"http://localhost:{restful_api_port}/mii/{deployment}"
    params = {"request": query}
    json_params = json.dumps(params)
    result = requests.post(url,
                           data=json_params,
                           headers={"Content-Type": "application/json"})
    assert result.status_code == 200
    assert "response" in result.json()


@pytest.mark.parametrize("load_with_sys_mem", [True])
def test_load_to_sys_mem(deployment, query):
    generator = mii.mii_query_handle(deployment)
    result = generator.query(query)
    assert result


@pytest.mark.parametrize("replica_num", [2])
def test_replicas(deployment, query, replica_num):
    generator = mii.mii_query_handle(deployment)
    # Replicas are given queries in round-robin, so test each model is responding
    for _ in range(replica_num):
        result = generator.query(query)
        assert result


@pytest.mark.deepspeed
@pytest.mark.parametrize("enable_deepspeed", [False])
@pytest.mark.parametrize("enable_zero", [True])
@pytest.mark.parametrize(
    "ds_config",
    [
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
    ],
)
def test_zero_config(deployment, query):
    generator = mii.mii_query_handle(deployment)
    result = generator.query(query)
    assert result


@pytest.mark.deepspeed
@pytest.mark.parametrize("expected_failure", [pydantic_v1.ValidationError])
@pytest.mark.parametrize(
    "enable_deepspeed, enable_zero, dtype",
    [(True,
      True,
      "fp32"),
     (False,
      True,
      "fp16")],
)
@pytest.mark.parametrize(
    "ds_config",
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
    ],
)
def test_zero_config_fail(deployment, query):
    assert "assertion_error" in str(deployment.value)
