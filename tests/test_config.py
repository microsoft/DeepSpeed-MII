# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

import mii


@pytest.mark.parametrize("replica_num", [2])
@pytest.mark.parametrize("tensor_parallel", [2])
@pytest.mark.parametrize(
    "device_map",
    [
        {
            "host_0": [[0,
                        1,
                        2,
                        3]]
        },
        {
            "host_0": [[0,
                        1]],
            "host_1": [[0]]
        },
        {
            "host_0": [[0,
                        1],
                       [2,
                        3],
                       [4,
                        5]]
        },
        {
            "host_0": [[0,
                        1]]
        },
    ],
)
@pytest.mark.parametrize("hostfile_content", [["host_0 slots=8", "host_1 slots=8"]])
def test_deploy_map_fail(mii_config):
    mii_config = mii.config.MIIConfig(**mii_config)
    with pytest.raises(ValueError):
        mii_config.generate_replica_configs()


@pytest.mark.parametrize("replica_num", [2])
@pytest.mark.parametrize("tensor_parallel", [2])
@pytest.mark.parametrize(
    "device_map",
    [
        {
            "host_0": [[0,
                        1],
                       [2,
                        3]]
        },
        {
            "host_0": [[0,
                        1]],
            "host_1": [[0,
                        1]]
        },
    ],
)
@pytest.mark.parametrize("hostfile_content", [["host_0 slots=4", "host_1 slots=4"]])
def test_deploy_map(mii_config):
    mii_config = mii.config.MIIConfig(**mii_config)
    mii_config.generate_replica_configs()


@pytest.mark.parametrize("replica_num", [2])
@pytest.mark.parametrize("tensor_parallel", [2])
@pytest.mark.parametrize(
    "hostfile_content",
    [["host_0 slots=4"],
     ["host_0 slots=2",
      "host_1 slots=2"],
     ["host_0 slots=8"]],
)
def test_auto_fill_deploy_map(mii_config):
    mii_config = mii.config.MIIConfig(**mii_config)
    mii_config.generate_replica_configs()


@pytest.mark.parametrize("device_map", [{"host_0": [[0, 1]]}, [[0, 1]], [0, 1], 1])
def test_deploy_map_input_types(mii_config):
    mii_config = mii.config.MIIConfig(**mii_config)
