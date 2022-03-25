'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import urllib.request
import os
import enum
import mii
from mii.utils import logger

try:
    from azureml.core import Environment
    from azureml.core.model import InferenceConfig
    from azureml.core.webservice import LocalWebservice
    from azureml.core.model import Model
except ImportError:
    azureml = None


#TODO naming..
class DeploymentType(enum.Enum):
    LOCAL = 1
    #expose GPUs
    LOCAL_AML = 2
    AML_ON_AKS = 3


ENV_NAME = "MII-Image-CUDA-11.3-grpc"


#TODO support this properly
def supported_model_list():
    return ["gpt2"]


def deploy(model_name,
           deployment_type,
           local_model_path=None,
           aml_model_tag=None,
           aml_deployment_name=None,
           aml_workspace=None,
           aks_target=None,
           aks_deploy_config=None,
           parallelism_config=None):

    assert model_name in supported_model_list(), f"Model {model_name} is not supported"

    if deployment_type == DeploymentType.LOCAL:
        return _deploy_local(model_name,
                             local_model_path=local_model_path,
                             parallelism_config=parallelism_config)

    if azureml is None:
        raise RuntimeError(
            "azureml is not available, please install via 'pip install mii[azure]' to get extra dependencies"
        )

    assert aml_workspace is not None, "Workspace cannot be none for AML deployments"

    if deployment_type == DeploymentType.LOCAL_AML:
        return _deploy_aml_local(model_name,
                                 aml_workspace,
                                 aml_deployment_name,
                                 aml_model_tag=aml_model_tag,
                                 parallelism_config=parallelism_config)

    assert aks_target is not None and aks_deploy_config is not None and aml_deployment_name is not None, "aks_target and aks_deployment_config must be provided for AML_ON_AKS deployment"

    return _deploy_aml_on_aks(model_name,
                              aml_workspace,
                              aks_target,
                              aks_deploy_config,
                              aml_deployment_name,
                              aml_model_tag=aml_model_tag,
                              parallelism_config=parallelism_config)


#TODO support this properly
#Fetch the specified model if it exists. If the model does not exist and its the default supported model type, create the model and register it
def _get_aml_model(model_name, aml_model_tag, aml_workspace):
    if model_name in aml_workspace.models.keys():
        return Model(workspace=aml_workspace,
                     name=model_name,
                     id=f"{model_name}:1",
                     version=1,
                     tags={},
                     properties={})
    else:
        assert False, "Model not found"


def _deploy_aml_on_aks(model_name,
                       aml_workspace,
                       aks_target,
                       aks_deploy_config,
                       aml_deployment_name,
                       aml_model_tag=None,
                       parallelism_config=None):
    global ENV_NAME
    model = _get_aml_model(model_name, aml_model_tag, aml_workspace)

    inference_config = InferenceConfig(
        environment=Environment.get(aml_workspace,
                                    name=ENV_NAME),
        #TODO make this more robust.
        #Use relative path from where we are running or an absolute path
        entry_script=os.path.join("models",
                                  model_name,
                                  "score.py"))

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


def _deploy_aml_local(model_name,
                      aml_workspace,
                      aml_deployment_name,
                      aml_model_tag=None,
                      parallelism_config=None):
    global ENV_NAME
    model = _get_aml_model(model_name, aml_model_tag, aml_workspace)

    inference_config = InferenceConfig(environment=Environment.get(aml_workspace,
                                                                   name=ENV_NAME),
                                       entry_script=os.path.join(
                                           "models",
                                           model_name,
                                           "score.py"))

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
    mii.import_score_file(model_name).init()
