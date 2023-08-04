# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import string
import os
import mii

from deepspeed.launcher.runner import fetch_hostfile

from .constants import DeploymentType, MII_MODEL_PATH_DEFAULT, MODEL_PROVIDER_MAP
from .utils import logger, get_task_name, get_provider_name
from .models.score import create_score_file
from .models import load_models
from .config import ReplicaConfig, LoadBalancerConfig


def support_legacy_api(task,
                       model,
                       deployment_type=DeploymentType.LOCAL,
                       model_path="",
                       enable_deepspeed=True,
                       enable_zero=False,
                       ds_config=None,
                       mii_config=None,
                       version=1):
    deployment_tag = deployment_name

    if ds_config is None:
        ds_config = {}
    if mii_config is None:
        mii_config = {}

    deployment_config = {
        "deployment_name": deployment_name,
        "task": task,
        "model": model,
        "model_path": model_path,
        "ds_optimize": enable_deepspeed,
        "ds_zero": enable_zero,
        "ds_config": ds_config,
    }
    for key, val in mii_config.items():
        if not hasattr(MIIConfig, key):
            deployment_config[key] = val
    deployments = [deployment_config]

    mii_config = {k: v for k, v in mii_config.items() if hasattr(MIIConfig, k)}
    mii_config["version"] = version
    mii_config["deployment_type"] = deployment_type

    return deployment_tag, deployments, mii_config


def deploy(deployment_name, deployment_config=None, mii_config=None, *args, **kwargs):
    if mii_config is None:
        mii_config = {}

    if args or kwargs:
        assert not deployment_config, "We do not support mixture of legacy and new API options, use latest API."
        kwargs["mii_config"] = mii_config
        deployment_config, mii_config = support_legacy_api(*args, **kwargs)

    deployment_config["deployment_name"] = deployment_name
    mii_config = mii.config.MIIConfig(**mii_config, deployment_config=deployment_config)

    if mii_config.deployment_config.enable_deepspeed:
        logger.info(
            f"************* MII is using DeepSpeed Optimizations to accelerate your model *************"
        )
    else:
        logger.info(
            f"************* DeepSpeed Optimizations not enabled. Please use enable_deepspeed to get better performance *************"
        )

    if deployment_type != DeploymentType.NON_PERSISTENT:
        create_score_file(mii_config)

    if deployment_type == DeploymentType.AML:
        _deploy_aml(mii_config)
    elif deployment_type == DeploymentType.LOCAL:
        return _deploy_local(mii_config)
    elif deployment_type == DeploymentType.NON_PERSISTENT:
        assert int(os.getenv('WORLD_SIZE', '1')) == mii_config.deployment_config.tensor_parallel, "World Size does not equal number of tensors. When using non-persistent deployment type, please launch with `deepspeed --num_gpus <tensor_parallel>`"
        deployment_name = mii_config.deployment_config.deployment_name
        model = mii_config.deployment_config.model
        task = mii_config.deployment_config.task
        model_path = mii_config.deployment_config.model_path
        enable_deepspeed = mii_config.deployment_config.enable_deepspeed
        enable_zero = mii_config.deployment_config.enable_zero
        provider = MODEL_PROVIDER_MAP[get_provider_name(model, task)]
        mii.non_persistent_models[deployment_name] = (load_models(
            task,
            model,
            model_path,
            enable_deepspeed,
            enable_zero,
            provider,
            mii_config),
                                                      task)
    else:
        raise Exception(f"Unknown deployment type: {deployment_type}")


def _deploy_local(mii_config):
    mii.utils.import_score_file(mii_config.deployment_config.deployment_name).init()


def _deploy_aml(mii_config):
    acr_name = mii.aml_related.utils.get_acr_name()
    mii.aml_related.utils.generate_aml_scripts(
        acr_name=acr_name,
        deployment_name=mii_config.deployment_config.deployment_name,
        model_name=mii_config.deployment_config.model,
        version=mii_config.version)
    print(
        f"AML deployment assets at {mii.aml_related.utils.aml_output_path(deployment_name)}"
    )
    print("Please run 'deploy.sh' to bring your deployment online")
