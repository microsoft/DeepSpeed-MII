'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import torch

import mii
from mii import utils
from mii.constants import DeploymentType
from mii.utils import logger, log_levels
from mii.models.utils import download_model_and_get_path
import pprint
import inspect


def create_score_file(deployment_name,
                      task,
                      model_name,
                      ds_optimize,
                      ds_zero,
                      ds_config,
                      mii_configs):
    config_dict = {}
    config_dict[mii.constants.TASK_NAME_KEY] = mii.get_task_name(task)
    config_dict[mii.constants.MODEL_NAME_KEY] = model_name
    config_dict[mii.constants.ENABLE_DEEPSPEED_KEY] = ds_optimize
    config_dict[mii.constants.MII_CONFIGS_KEY] = mii_configs.dict()
    config_dict[mii.constants.ENABLE_DEEPSPEED_ZERO_KEY] = ds_zero
    config_dict[mii.constants.DEEPSPEED_CONFIG_KEY] = ds_config

    if len(mii.__path__) > 1:
        logger.warning(
            f"Detected mii path as multiple sources: {mii.__path__}, might cause unknown behavior"
        )

    with open(os.path.join(mii.__path__[0], "models/generic_model/score.py"), "r") as fd:
        score_src = fd.read()

    # update score file w. global config dict
    source_with_config = utils.debug_score_preamble()
    source_with_config += f"{score_src}\n"
    source_with_config += f"configs = {pprint.pformat(config_dict,indent=4)}"

    with open(utils.generated_score_path(deployment_name), "w") as fd:
        fd.write(source_with_config)
        fd.write("\n")


def deploy(task,
           model,
           deployment_name,
           deployment_type=DeploymentType.LOCAL,
           local_model_path=None,
           enable_deepspeed=True,
           enable_zero=False,
           ds_config=None,
           mii_config={}):
    """Deploy a task using specified model. For usage examples see:

        mii/examples/local/text-generation-example.py


    Arguments:
        task: Name of the machine learning task to be deployed.Currently MII supports the following list of tasks
            ``['text-generation', 'question-answering']``

        model: Name of a supported model for the task. Models in MII are sourced from multiple open-source projects
            such as Huggingface Transformer, FairSeq, EluetherAI etc. For the list of supported models for each task, please
            see here [TODO].

        deployment_name: Name of the deployment. Used as an identifier for posting queries for ``LOCAL`` deployment.

        deployment_type: One of the ``enum mii.DeploymentTypes: [LOCAL]``.
            *``LOCAL`` uses a grpc server to create a local deployment, and query the model must be done by creating a query handle using
              `mii.mii_query_handle` and posting queries using ``mii_request_handle.query`` API,

        local_model_path: Optional: Local folder where the model checkpoints are available.
            This should be provided if you want to use your own checkpoint instead of the default open-source checkpoints for the supported models for `LOCAL` deployment.

        enable_deepspeed: Optional: Defaults to True. Use this flag to enable or disable DeepSpeed-Inference optimizations

        enable_zero: Optional: Defaults to False. Use this flag to enable or disable DeepSpeed-ZeRO inference

        ds_config: Optional: Defaults to None. Use this to specify the DeepSpeed configuration when enabling DeepSpeed-ZeRO inference

        force_register_model: Optional: Defaults to False. For AML deployments, set it to True if you want to re-register your model
            with the same ``aml_model_tags`` using checkpoints from ``local_model_path``.

        mii_config: Optional: Dictionary specifying optimization and deployment configurations that should override defaults in ``mii.config.MIIConfig``.
            mii_config is future looking to support extensions in optimization strategies supported by DeepSpeed Inference as we extend mii.
            As of now, it can be used to set tensor-slicing degree using 'tensor_parallel' and port number for deployment using 'port_number'.
    Returns:
        If deployment_type is `LOCAL`, returns just the name of the deployment that can be used to create a query handle using `mii.mii_query_handle(deployment_name)`

    """
    # parse and validate mii config
    mii_config = mii.config.MIIConfig(**mii_config)
    if enable_zero:
        if ds_config.get("fp16", {}).get("enabled", False):
            assert (mii_config.torch_dtype() == torch.half), "MII Config Error: MII dtype and ZeRO dtype must match"
        else:
            assert (mii_config.torch_dtype() == torch.float), "MII Config Error: MII dtype and ZeRO dtype must match"
    assert not (enable_deepspeed and enable_zero), "MII Config Error: DeepSpeed and ZeRO cannot both be enabled, select only one"

    task = mii.get_task(task)
    mii.check_if_task_and_model_is_valid(task, model)
    if enable_deepspeed:
        mii.check_if_task_and_model_is_supported(task, model)

    logger.info(f"*************DeepSpeed Optimizations: {enable_deepspeed}*************")

    create_score_file(deployment_name,
                      task,
                      model,
                      enable_deepspeed,
                      enable_zero,
                      ds_config,
                      mii_config)

    if deployment_type == DeploymentType.AML:
        print(f"Score file created at {utils.generated_score_path(deployment_name)}")
    elif deployment_type == DeploymentType.LOCAL:
        return _deploy_local(deployment_name, local_model_path=local_model_path)
    else:
        raise Exception(f"Unknown deployment type: {deployment_type}")


def _deploy_local(deployment_name, local_model_path=None):
    mii.set_model_path(local_model_path)
    mii.import_score_file(deployment_name).init()
