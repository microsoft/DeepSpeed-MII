# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from typing import Optional, Any, Dict, Tuple

import mii
from mii.backend import MIIClient  # , MIIServer
from mii.batching import MIIPipeline, MIIAsyncPipeline
from mii.config import get_mii_config, ModelConfig, MIIConfig
from mii.constants import DeploymentType
from mii.errors import UnknownArgument
from mii.modeling.models import load_model
from mii.score import create_score_file
from mii.modeling.tokenizers import load_tokenizer
from mii.utils import import_score_file


def _parse_kwargs_to_model_config(
    model_name_or_path: str = "",
    model_config: Optional[Dict] = None,
    **kwargs,
) -> Tuple[ModelConfig,
           Dict[str,
                Any]]:
    if model_config is None:
        model_config = {}

    assert isinstance(model_config, dict), "model_config must be a dict"

    # If model_name_or_path is set in model config, make sure it matches the kwarg
    if model_name_or_path:
        if "model_name_or_path" in model_config:
            assert (
                model_config.get("model_name_or_path") == model_name_or_path
            ), "model_name_or_path in model_config must match model_name_or_path"
        model_config["model_name_or_path"] = model_name_or_path

    # Fill model_config dict with relevant kwargs, store remaining kwargs in a new dict
    remaining_kwargs = {}
    for key, val in kwargs.items():
        if key in ModelConfig.__dict__["__fields__"]:
            if key in model_config:
                assert (
                    model_config.get(key) == val
                ), f"{key} in model_config must match {key}"
            model_config[key] = val
        else:
            remaining_kwargs[key] = val

    # Create the ModelConfig object and return it with remaining kwargs
    model_config = ModelConfig(**model_config)

    return model_config, remaining_kwargs


def _parse_kwargs_to_mii_config(
    model_name_or_path: str = "",
    model_config: Optional[Dict] = None,
    mii_config: Optional[Dict] = None,
    **kwargs,
) -> MIIConfig:
    if mii_config is None:
        mii_config = {}

    if model_config is None:
        model_config = mii_config.get("model_config", {})

    # Parse all model_config kwargs
    model_config, remaining_kwargs = _parse_kwargs_to_model_config(
        model_name_or_path=model_name_or_path, model_config=model_config, **kwargs
    )

    assert isinstance(mii_config, dict), "mii_config must be a dict"

    mii_config["model_config"] = model_config

    # Fill mii_config dict with relevant kwargs, raise error on unknown kwargs
    for key, val in remaining_kwargs.items():
        if key in MIIConfig.__dict__["__fields__"]:
            if key in mii_config:
                assert (
                    mii_config.get(key) == val
                ), f"{key} in mii_config must match {key}"
            mii_config[key] = val
        else:
            raise UnknownArgument(f"Keyword argument {key} not recognized")

    # Return the MIIConfig object
    mii_config = MIIConfig(**mii_config)
    return mii_config


def client(model_or_deployment_name: str) -> MIIClient:
    """
    Creates a client object for interfacing with an existing persistent model deployment.

    :param model_or_deployment_name: Name of the HuggingFace model name for the
        persistent model deployment. If `deployment_name` was provided input to
        :func:`mii.serve`, users should provide the `deployment_name` string instead.

    :raises UnknownArgument: Raised when provided keyword argument does not
        match any field in :class:`ModelConfig <mii.config.ModelConfig>` or
        :class:`MIIConfig <mii.config.MIIConfig>`.

    :return: Client object that can be used to interface with the deployed
        persistent model server, which uses ragged batching and dynamic splitfuse.
    """
    mii_config = get_mii_config(model_or_deployment_name)

    return MIIClient(mii_config)


def serve(
    model_name_or_path: str = "",
    model_config: Optional[Dict] = None,
    mii_config: Optional[Dict] = None,
    **kwargs,
) -> MIIClient:
    """
    Creates a persistent MII model deployment from a locally stored model path
    or HuggingFace model name.

    :param model_name_or_path: HuggingFace model name or path to locally stored
        model. This must be provided here or in the `model_config` dictionary.
    :param model_config: Dictionary containing model configuration fields. See
        :class:`ModelConfig <mii.config.ModelConfig>` for a full list of options.
        Users can pass these options in a dictionary here or as keyword arguments to
        the function.
    :param mii_config: Dictionary containing DeepSpeed-MII configuration fields.
        See :class:`MIIConfig <mii.config.MIIConfig>` for a full list of options.
        Users can pass these options in a dictionary here or as keyword arguments to
        the function.

    :raises UnknownArgument: Raised when provided keyword argument does not
        match any field in :class:`ModelConfig <mii.config.ModelConfig>` or
        :class:`MIIConfig <mii.config.MIIConfig>`.

    :return: Client object that can be used to interface with the deployed
        persistent model server, which uses ragged batching and dynamic splitfuse.
    """
    mii_config = _parse_kwargs_to_mii_config(
        model_name_or_path=model_name_or_path,
        model_config=model_config,
        mii_config=mii_config,
        **kwargs,
    )

    # Eventually we will move away from generating score files, leaving this
    # here as a placeholder for now.
    # MIIServer(mii_config)
    create_score_file(mii_config)

    if mii_config.deployment_type == DeploymentType.LOCAL:
        import_score_file(mii_config.deployment_name, DeploymentType.LOCAL).init()
        return MIIClient(mii_config=mii_config)
    if mii_config.deployment_type == DeploymentType.AML:
        acr_name = mii.aml_related.utils.get_acr_name()
        mii.aml_related.utils.generate_aml_scripts(
            acr_name=acr_name,
            deployment_name=mii_config.deployment_name,
            model_name=mii_config.model_config.model,
            task_name=mii_config.model_config.task,
            replica_num=mii_config.model_config.replica_num,
            instance_type=mii_config.instance_type,
            version=mii_config.version,
        )
        print(
            f"AML deployment assets at {mii.aml_related.utils.aml_output_path(mii_config.deployment_name)}"
        )
        print("Please run 'deploy.sh' to bring your deployment online")


def pipeline(
    model_name_or_path: str = "",
    model_config: Optional[Dict] = None,
    all_rank_output: bool = False,
    **kwargs,
) -> MIIPipeline:
    """
    Creates a non-persistent MII model pipeline from a locally stored model path
    or HuggingFace model name.

    :param model_name_or_path: HuggingFace model name or path to locally stored
        model. This must be provided here or in the `model_config` dictionary.
    :param model_config: Dictionary containing model configuration fields. See
        :class:`ModelConfig <mii.config.ModelConfig>` for a full list of options.
        Users can pass these options in a dictionary here or as keyword arguments to
        the function.
    :param all_rank_output: Whether to return generated text on all ranks
        (when using `tensor_parallel>1`). If `True`, all ranks will return the
        same output. If `False`, only rank 0 will return output and the rest
        will return `None`.

    :raises UnknownArgument: Raised when provided keyword argument does not
        match any field in :class:`ModelConfig <mii.config.ModelConfig>`.

    :return: Non-persistent model pipeline using ragged batching and dynamic splitfuse.
    """
    model_config, remaining_kwargs = _parse_kwargs_to_model_config(
        model_name_or_path=model_name_or_path, model_config=model_config, **kwargs
    )
    if remaining_kwargs:
        raise UnknownArgument(
            f"Keyword argument(s) {remaining_kwargs.keys()} not recognized")

    inference_engine = load_model(model_config)
    tokenizer = load_tokenizer(model_config)
    inference_pipeline = MIIPipeline(
        inference_engine=inference_engine,
        tokenizer=tokenizer,
        model_config=model_config,
        all_rank_output=all_rank_output,
    )
    return inference_pipeline


def async_pipeline(model_config: ModelConfig) -> MIIAsyncPipeline:
    inference_engine = load_model(model_config)
    tokenizer = load_tokenizer(model_config)
    inference_pipeline = MIIAsyncPipeline(
        inference_engine=inference_engine,
        tokenizer=tokenizer,
        model_config=model_config,
    )
    return inference_pipeline
