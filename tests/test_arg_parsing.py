# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from mii.api import _parse_kwargs_to_model_config, _parse_kwargs_to_mii_config
from mii.errors import UnknownArgument


def test_model_name_or_path():
    # model_name_or_path is required
    with pytest.raises(ValueError):
        _parse_kwargs_to_mii_config()
    with pytest.raises(ValueError):
        _parse_kwargs_to_model_config()

    # passing model_name_or_path as positional arg
    mii_config = _parse_kwargs_to_mii_config("test")
    assert mii_config.model_config.model_name_or_path == "test"
    model_config, _ = _parse_kwargs_to_model_config("test")
    assert model_config.model_name_or_path == "test"

    # passing model_name_or_path in model_config
    mii_config = _parse_kwargs_to_mii_config(model_config={"model_name_or_path": "test"})
    assert mii_config.model_config.model_name_or_path == "test"
    mii_config = _parse_kwargs_to_mii_config(
        mii_config={"model_config": {
            "model_name_or_path": "test"
        }})
    assert mii_config.model_config.model_name_or_path == "test"
    model_config, _ = _parse_kwargs_to_model_config(
        model_config={"model_name_or_path": "test"}
    )
    assert model_config.model_name_or_path == "test"

    # checking that model_name_or_path in model_config matches positional arg
    with pytest.raises(AssertionError):
        _parse_kwargs_to_mii_config("test", model_config={"model_name_or_path": "test2"})
    with pytest.raises(AssertionError):
        _parse_kwargs_to_mii_config(
            "test",
            mii_config={"model_config": {
                "model_name_or_path": "test2"
            }})
    with pytest.raises(AssertionError):
        _parse_kwargs_to_model_config("test",
                                      model_config={"model_name_or_path": "test2"})


def test_only_kwargs():
    mii_config = _parse_kwargs_to_mii_config("test",
                                             tensor_parallel=2,
                                             enable_restful_api=True)
    assert mii_config.model_config.model_name_or_path == "test"
    assert mii_config.model_config.tensor_parallel == 2
    assert mii_config.enable_restful_api is True

    model_config, _ = _parse_kwargs_to_model_config("test", tensor_parallel=2)
    assert model_config.model_name_or_path == "test"
    assert model_config.tensor_parallel == 2


def test_only_config_dicts():
    mii_config = _parse_kwargs_to_mii_config(
        mii_config={"enable_restful_api": True},
        model_config={
            "model_name_or_path": "test",
            "tensor_parallel": 2
        },
    )
    assert mii_config.model_config.model_name_or_path == "test"
    assert mii_config.model_config.tensor_parallel == 2
    assert mii_config.enable_restful_api is True

    mii_config = _parse_kwargs_to_mii_config(
        mii_config={
            "enable_restful_api": True,
            "model_config": {
                "model_name_or_path": "test",
                "tensor_parallel": 2
            },
        })
    assert mii_config.model_config.model_name_or_path == "test"
    assert mii_config.model_config.tensor_parallel == 2
    assert mii_config.enable_restful_api is True

    model_config, _ = _parse_kwargs_to_model_config(
        model_config={"model_name_or_path": "test", "tensor_parallel": 2}
    )
    assert model_config.model_name_or_path == "test"
    assert model_config.tensor_parallel == 2


def test_unknown_kwargs():
    with pytest.raises(UnknownArgument):
        _parse_kwargs_to_mii_config("test", unknown_kwarg=True)

    _, remaining_kwargs = _parse_kwargs_to_model_config("test", unknown_kwarg=True)
    assert remaining_kwargs == {"unknown_kwarg": True}
