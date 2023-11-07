# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import time
import torch
import os
import mii
from types import SimpleNamespace


@pytest.fixture(scope="function", params=[None])
def tensor_parallel(request):
    if request.param is not None:
        return request.param
    return int(os.getenv("WORLD_SIZE", "1"))


@pytest.fixture(scope="function", params=[50050])
def port_number(request):
    return request.param


@pytest.fixture(scope="function", params=[1])
def replica_num(request):
    return request.param


@pytest.fixture(scope="function", params=[False])
def enable_restful_api(request):
    return request.param


@pytest.fixture(scope="function", params=[28080])
def restful_api_port(request):
    return request.param


@pytest.fixture(scope="function", params=[mii.TaskType.TEXT_GENERATION])
def task_name(request):
    return request.param


@pytest.fixture(scope="function", params=["facebook/opt-1.3b"])
def model_name(request):
    return request.param


@pytest.fixture(scope="function", params=[mii.DeploymentType.LOCAL])
def deployment_type(request):
    return request.param


@pytest.fixture(scope="function", params=[True])
def all_rank_output(request):
    return request.param


@pytest.fixture(scope="function")
def model_config(
    task_name: str,
    model_name: str,
    tensor_parallel: int,
    replica_num: int,
    all_rank_output: bool,
):
    config = SimpleNamespace(
        model_name_or_path=model_name,
        task=task_name,
        tensor_parallel=tensor_parallel,
        replica_num=replica_num,
        all_rank_output=all_rank_output,
    )
    return config.__dict__


@pytest.fixture(scope="function")
def mii_config(
    deployment_type: str,
    port_number: int,
    enable_restful_api: bool,
    restful_api_port: int,
):
    config = SimpleNamespace(
        deployment_type=deployment_type,
        port_number=port_number,
        enable_restful_api=enable_restful_api,
        restful_api_port=restful_api_port,
    )
    return config.__dict__


@pytest.fixture(scope="function", params=[None], ids=["nofail"])
def expected_failure(request):
    return request.param


@pytest.fixture(scope="function")
def pipeline(model_config, expected_failure):
    if expected_failure is not None:
        with pytest.raises(expected_failure) as excinfo:
            mii.pipeline(model_config=model_config)
        yield excinfo
    else:
        pipe = mii.pipeline(model_config=model_config)
        yield pipe
        del pipe.inference_engine
        del pipe
        torch.cuda.empty_cache()


@pytest.fixture(scope="function")
def deployment(mii_config, model_config, expected_failure):
    if expected_failure is not None:
        with pytest.raises(expected_failure) as excinfo:
            mii.serve(model_config=model_config, mii_config=mii_config)
        yield excinfo
    else:
        client = mii.serve(model_config=model_config, mii_config=mii_config)
        yield client
        client.terminate_server()
        time.sleep(1)


@pytest.fixture(scope="function", params=["DeepSpeed is the greatest"], ids=["query0"])
def query(request):
    return request.param
