'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import json

import mii
from mii import utils
from mii.constants import DeploymentType, MII_PARALLELISM_DEFAULT
from mii.utils import logger, log_levels
from mii.models.utils import download_model_and_get_path
import pprint
try:
    from azureml.core import Environment
    from azureml.core.model import InferenceConfig
    from azureml.core.webservice import LocalWebservice
    from azureml.core.model import Model
except ImportError:
    azureml_available = False
else:
    azureml_available = True

ENV_NAME = "MII-Image"


def create_score_file(task, model_name, ds_optimize, parallelism_config):
    config_dict = {}
    config_dict[mii.constants.TASK_NAME_KEY] = mii.get_task_name(task)
    config_dict[mii.constants.MODEL_NAME_KEY] = model_name
    config_dict[mii.constants.ENABLE_DEEPSPEED_KEY] = ds_optimize
    config_dict[mii.constants.PARALLELISM_KEY] = parallelism_config

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

    with open(utils.generated_score_path(), "w") as fd:
        fd.write(source_with_config)
        fd.write("\n")


#Fetch the specified model if it exists. If the model does not exist and its the default supported model type, create the model and register it
def _get_aml_model(task_name,
                   model_name,
                   aml_model_tags,
                   aml_workspace,
                   local_model_path=None,
                   force_register=False):

    models = Model.list(workspace=aml_workspace,
                        name=model_name,
                        tags=[[str(key),
                               str(value)] for key,
                              value in aml_model_tags.items()])

    if len(models) == 0 or force_register:

        if len(models) > 0:
            logger.warning(
                f"{model_name} with {aml_model_tags} is already registered, but registering a new version due to force_register: {force_register}"
            )

        model_path = download_model_and_get_path(
            task_name,
            model_name) if local_model_path is None else local_model_path

        logger.info(
            f"Registering {model_name} model with tag {aml_model_tags} from path {model_path}"
        )
        return Model.register(workspace=aml_workspace,
                              model_path=model_path,
                              model_name=model_name,
                              tags=aml_model_tags)

    else:
        logger.info(
            f"Pre-registered model {model_name} with tag {aml_model_tags} found. Returning existing model."
        )
        return Model(workspace=aml_workspace, name=model_name, tags=aml_model_tags)


def _get_inference_config(aml_workspace):
    global ENV_NAME

    environment = None
    if ENV_NAME in Environment.list(aml_workspace).keys():
        logger.info(
            f"Pre-registered Environment: {ENV_NAME} found. Fetching this environment.")

        environment = Environment.get(aml_workspace, name=ENV_NAME)
    else:
        logger.info(
            f"Environment: {ENV_NAME} has not been registered. Creating image and registering it to your aml workspace."
        )

        env = Environment.from_dockerfile(
            name=ENV_NAME,
            dockerfile=os.path.join(mii.__path__[0],
                                    "aml_environment/Dockerfile"),
            pip_requirements=os.path.join(mii.__path__[0],
                                          'aml_environment/requirements.txt'))
        env.register(aml_workspace)

    inference_config = InferenceConfig(environment=Environment.get(aml_workspace,
                                                                   name=ENV_NAME),
                                       entry_script=utils.generated_score_path())
    return inference_config


def deploy(task_name,
           model_name,
           deployment_type,
           local_model_path=None,
           aml_model_tags=None,
           force_register_model=False,
           aml_deployment_name=None,
           aml_workspace=None,
           aks_target=None,
           aks_deploy_config=None,
           enable_deepspeed=True,
           parallelism_config=MII_PARALLELISM_DEFAULT):

    task = mii.get_task(task_name)
    mii.check_if_task_and_model_is_supported(task, model_name)

    logger.info(f"*************DeepSpeed Optimizations: {enable_deepspeed}*************")

    create_score_file(task, model_name, enable_deepspeed, parallelism_config)

    if deployment_type == DeploymentType.LOCAL:
        return _deploy_local(model_name,
                             local_model_path=local_model_path,
                             parallelism_config=parallelism_config)

    if not azureml_available:
        raise RuntimeError(
            "azureml is not available, please install via 'pip install mii[azure]' to get extra dependencies"
        )

    assert aml_workspace is not None, "Workspace cannot be none for AML deployments"
    assert aml_deployment_name is not None, "Must provide aml_deployment_name for AML deployments"

    #either return a previously registered model, or register a new model
    model = _get_aml_model(task_name,
                           model_name,
                           aml_model_tags,
                           aml_workspace,
                           local_model_path=local_model_path,
                           force_register=force_register_model)

    logger.info(f"Deploying model {model}")

    #return
    inference_config = _get_inference_config(aml_workspace)

    if deployment_type == DeploymentType.AML_LOCAL:
        return _deploy_aml_local(model,
                                 inference_config,
                                 aml_workspace,
                                 aml_deployment_name,
                                 parallelism_config=parallelism_config)

    assert aks_target is not None and aks_deploy_config is not None and aml_deployment_name is not None, "aks_target and aks_deployment_config must be provided for AML_ON_AKS deployment"

    return _deploy_aml_on_aks(model,
                              inference_config,
                              aml_workspace,
                              aks_target,
                              aks_deploy_config,
                              aml_deployment_name,
                              parallelism_config=parallelism_config)


def _deploy_aml_on_aks(model,
                       inference_config,
                       aml_workspace,
                       aks_target,
                       aks_deploy_config,
                       aml_deployment_name,
                       parallelism_config=None):

    service = Model.deploy(
        aml_workspace,
        aml_deployment_name,
        [model],
        inference_config=inference_config,
        deployment_config=aks_deploy_config,
        deployment_target=aks_target,
        overwrite=True,
    )

    service.wait_for_deployment(show_output=True)

    return service


def _deploy_aml_local(model,
                      inference_config,
                      aml_workspace,
                      aml_deployment_name,
                      parallelism_config=None):

    deployment_config = LocalWebservice.deploy_configuration(port=6789)

    service = Model.deploy(
        aml_workspace,
        aml_deployment_name,
        [model],
        inference_config=inference_config,
        deployment_config=deployment_config,
        overwrite=True,
    )

    service.wait_for_deployment(show_output=True)

    return service


def _deploy_local(model_name, local_model_path=None, parallelism_config=None):
    mii.set_model_path(local_model_path)
    mii.import_score_file().init()
