'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os

import mii
from mii import utils
from mii.constants import DeploymentType
from mii.utils import logger, log_levels
from mii.models.utils import download_model_and_get_path
import pprint
import inspect


def create_score_file(deployment_name, task, model_name, ds_optimize, mii_configs):
    config_dict = {}
    config_dict[mii.constants.TASK_NAME_KEY] = mii.get_task_name(task)
    config_dict[mii.constants.MODEL_NAME_KEY] = model_name
    config_dict[mii.constants.ENABLE_DEEPSPEED_KEY] = ds_optimize
    config_dict[mii.constants.MII_CONFIGS_KEY] = mii_configs.dict()

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


def deploy(task_name,
           model_name,
           deployment_type,
           deployment_name,
           local_model_path=None,
           enable_deepspeed=True,
           mii_configs={}):
    """Deploy a task using specified model. For usage examples see:

        mii/examples/local/text-generation-example.py


    Arguments:
        task_name: Name of the machine learning task to be deployed.Currently MII supports the following list of tasks
            ``['text-generation', 'question-answering']``

        model_name: Name of a supported model for the task. Models in MII are sourced from multiple open-source projects
            such as Huggingface Transformer, FairSeq, EluetherAI etc. For the list of supported models for each task, please
            see here [TODO].

        deployment_type: One of the ``enum mii.DeploymentTypes: [LOCAL]``.
            *``LOCAL`` uses a grpc server to create a local deployment, and query the model must be done by creating a query handle using
              `mii.mii_query_handle` and posting queries using ``mii_request_handle.query`` API,

        deployment_name: Name of the deployment. Used as an identifier for posting queries for ``LOCAL`` deployment.

        local_model_path: Optional: Local folder where the model checkpoints are available.
            This should be provided if you want to use your own checkpoint instead of the default open-source checkpoints for the supported models for `LOCAL` deployment.

        enable_deepspeed: Optional: Defaults to True. Use this flag to enable or disable DeepSpeed-Inference optimizations

        mii_configs: Optional: Dictionary specifying optimization and deployment configurations that should override defaults in ``mii.config.MIIConfig``.
            mii_config is future looking to support extensions in optimization strategies supported by DeepSpeed Inference as we extend mii.
            As of now, it can be used to set tensor-slicing degree using 'tensor_parallel' and port number for deployment using 'port_number'.
    Returns:
        If deployment_type is `LOCAL`, returns just the name of the deployment that can be used to create a query handle using `mii.mii_query_handle(deployment_name)`

    """
    # parse and validate mii config
    mii_configs = mii.config.MIIConfig(**mii_configs)

    task = mii.get_task(task_name)
    mii.check_if_task_and_model_is_supported(task, model_name)

    logger.info(f"*************DeepSpeed Optimizations: {enable_deepspeed}*************")

    create_score_file(deployment_name, task, model_name, enable_deepspeed, mii_configs)

    assert deployment_type == DeploymentType.LOCAL, "MII currently supports only local deployment"
    return _deploy_local(deployment_name, local_model_path=local_model_path)


def _deploy_local(deployment_name, local_model_path=None):
    mii.set_model_path(local_model_path)
    mii.import_score_file(deployment_name).init()
