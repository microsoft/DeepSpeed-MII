# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import pydantic

import mii


@pytest.mark.parametrize("port_number", [12345])
@pytest.mark.parametrize("tensor_parallel", [4])
def test_base_configs(deployment_name, mii_config, deployment_config):
    deployment_config["deployment_name"] = deployment_name
    mii_config["deployment_config"] = deployment_config
    mii_config = mii.config.MIIConfig(**mii_config)

    assert mii_config.port_number == 12345
    assert mii_config.deployment_config.tensor_parallel == 4


@pytest.mark.parametrize("port_number", ["fail"])
@pytest.mark.parametrize("tensor_parallel", [3.5])
def test_base_configs_literalfail(deployment_name, mii_config, deployment_config):
    with pytest.raises(pydantic.ValidationError):
        deployment_config["deployment_name"] = deployment_name
        mii_config["deployment_config"] = deployment_config
        mii_config = mii.config.MIIConfig(**mii_config)
