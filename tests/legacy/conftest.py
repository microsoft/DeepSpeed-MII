# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import os
import mii.legacy as mii
from types import SimpleNamespace


@pytest.fixture(scope="function", params=["fp16"])
def dtype(request):
    return request.param


@pytest.fixture(scope="function", params=[1])
def tensor_parallel(request):
    return request.param


@pytest.fixture(scope="function", params=[50050])
def port_number(request):
    return request.param


@pytest.fixture(scope="function", params=[False])
def meta_tensor(request):
    return request.param


@pytest.fixture(scope="function", params=[False])
def load_with_sys_mem(request):
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


@pytest.fixture(scope="function", params=["text-generation"])
def task_name(request):
    return request.param


@pytest.fixture(scope="function", params=["bigscience/bloom-560m"])
def model_name(request):
    return request.param


@pytest.fixture(scope="function")
def deployment_name(model_name):
    return model_name + "-deployment"


@pytest.fixture(scope="function", params=[mii.DeploymentType.LOCAL])
def deployment_type(request):
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


@pytest.fixture(scope="function")
def replace_with_kernel_inject(model_name):
    if "clip-vit" in model_name:
        return False
    return True


@pytest.fixture(scope="function")
def model_config(
    task_name: str,
    model_name: str,
    dtype: str,
    tensor_parallel: int,
    meta_tensor: bool,
    load_with_sys_mem: bool,
    replica_num: int,
    enable_deepspeed: bool,
    enable_zero: bool,
    ds_config: dict,
    replace_with_kernel_inject: bool,
):
    config = SimpleNamespace(
        skip_model_check=True, # TODO: remove this once conversation task check is fixed
        task=task_name,
        model=model_name,
        dtype=dtype,
        tensor_parallel=tensor_parallel,
        model_path=os.getenv("TRANSFORMERS_CACHE",
                             ""),
        meta_tensor=meta_tensor,
        load_with_sys_mem=load_with_sys_mem,
        replica_num=replica_num,
        enable_deepspeed=enable_deepspeed,
        enable_zero=enable_zero,
        ds_config=ds_config,
        replace_with_kernel_inject=replace_with_kernel_inject,
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


@pytest.fixture(scope="function", params=[None])
def expected_failure(request):
    return request.param


@pytest.fixture(scope="function")
def deployment(deployment_name, mii_config, model_config, expected_failure):
    if expected_failure is not None:
        with pytest.raises(expected_failure) as excinfo:
            mii.deploy(
                deployment_name=deployment_name,
                mii_config=mii_config,
                model_config=model_config,
            )
        yield excinfo
    else:
        mii.deploy(
            deployment_name=deployment_name,
            mii_config=mii_config,
            model_config=model_config,
        )
        yield deployment_name
        mii.terminate(deployment_name)


@pytest.fixture(scope="function", params=[{"query": "DeepSpeed is the greatest"}])
def query(request):
    return request.param
