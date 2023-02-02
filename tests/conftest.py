import pytest
import os
from types import SimpleNamespace

import mii


def validate_config(config):
    if (config.model in ['bert-base-uncased']) and (config.mii_config['dtype']
                                                    == 'fp16'):
        pytest.skip(f"Model f{config.model} not supported for FP16")
    elif config.mii_config['dtype'] == "fp32" and "bloom" in config.model:
        pytest.skip('bloom does not support fp32')


''' These fixtures provide default values for the deployment config '''


@pytest.fixture(scope="function", params=['fp32'])
def dtype(request):
    return request.param


@pytest.fixture(scope="function", params=[1])
def tensor_parallel(request):
    return request.param


@pytest.fixture(scope="function", params=[50050])
def port_number(request):
    return request.param


@pytest.fixture(scope="function", params=[5001])
def local_aml_port(request):
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


@pytest.fixture(scope="function")
def mii_configs(dtype: str,
                tensor_parallel: int,
                port_number: int,
                load_with_sys_mem: bool):
    return {
        'dtype': dtype,
        'tensor_parallel': tensor_parallel,
        'port_number': port_number,
        'load_with_sys_mem': load_with_sys_mem,
        'local_aml_port': local_aml_port,
    }


@pytest.fixture(scope="function", params=[None])
def expected_failure(request):
    return request.param


@pytest.fixture(scope="function")
def local_deployment_config(task_name: str,
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


@pytest.fixture(scope="function")
def local_deployment(local_deployment_config, expected_failure):
    if expected_failure is not None:
        with pytest.raises(expected_failure) as excinfo:
            mii.deploy(**local_deployment_config.__dict__)
        yield excinfo
    else:
        mii.deploy(**local_deployment_config.__dict__)
        yield local_deployment_config
        mii.terminate(local_deployment_config.deployment_name)


@pytest.fixture(scope="function")
def aml_local_deployment_config(task_name: str,
                      model_name: str,
                      mii_configs: dict,
                      enable_deepspeed: bool,
                      enable_zero: bool,
                      ds_config: dict):
    config = SimpleNamespace(task=task_name,
                             model=model_name,
                             deployment_type=mii.DeploymentType.AML_LOCAL,
                             deployment_name=model_name + "-deployment",
                             model_path=os.getenv("TRANSFORMERS_CACHE",
                                                  None),
                             mii_config=mii_configs,
                             enable_deepspeed=enable_deepspeed,
                             enable_zero=enable_zero,
                             ds_config=ds_config)
    validate_config(config)
    return config


@pytest.fixture(scope="function")
def aml_local_deployment(aml_local_deployment_config, expected_failure):
    if expected_failure is not None:
        with pytest.raises(expected_failure) as excinfo:
            mii.deploy(**aml_local_deployment_config.__dict__)
        yield excinfo
    else:
        mii.deploy(**aml_local_deployment_config.__dict__)
        yield aml_local_deployment_config
        mii.terminate(aml_local_deployment_config.deployment_name)
