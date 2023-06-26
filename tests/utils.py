# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import pytest


def validate_config(config):
    if (config.model in ['bert-base-uncased']) and (config.mii_config['dtype']
                                                    == 'fp16'):
        pytest.skip(f"Model f{config.model} not supported for FP16")
    elif config.mii_config['dtype'] == "fp32" and "bloom" in config.model:
        pytest.skip('bloom does not support fp32')


''' These fixtures provide default values for the deployment config '''


@pytest.fixture(scope="function", params=[False])
def meta_tensor(request):
    return request.param


@pytest.fixture(scope="function", params=['fp16'])
def dtype(request):
    return request.param


@pytest.fixture(scope="function", params=[1])
def tensor_parallel(request):
    return request.param


@pytest.fixture(scope="function", params=[False])
def load_with_sys_mem(request):
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


@pytest.fixture(scope="function", params=[None])
def expected_failure(request):
    return request.param
