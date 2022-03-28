'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import urllib.request
import os
import enum
import mii
from mii.utils import logger
import json

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


#TODO do this properly
def check_if_supported(task_name, model_name):
    assert task_name in ['text-generation', 'sequence-classification', 'question-answer'], "Not a supported task type"
    supported = False
    if task_name in ['text-generation']:
        supported_models = ['gpt2']
        assert model_name in supported_models, f"{task_name} only supports {supported_models}"
    elif task_name in ['sequence-classification']:
        supported_models = 'roberta-large-mnli'
        assert model_name in supported_models, f"{task_name} only supports {supported_models}"
    elif task_name in ['question-answering']:
        supported_models = ['deepset/roberta-large-squad2']
        assert model_name in supported_models, f"{task_name} only supports {supported_models}"
    else:
        assert False, "Does not support model {model_name} for task {task_name}"


#TODO do this properly
def create_score_file(task_name, model_name, parallelism_config):
    config_dict = {}
    config_dict['task_name'] = task_name
    config_dict['model_name'] = model_name
    config_dict['parallelism_config'] = parallelism_config
    config_to_append = json.dumps(config_dict)
    import subprocess
    subprocess.run(['pwd'])

    #open text file in read mode
    #TODO how to locate the absolute path of this file
    source_file_template = open("../mii/models/generic_model/score.py", "r")

    #read whole file to a string
    source_in_str = source_file_template.read()

    #close file
    source_file_template.close()

    source_with_config = source_in_str + "\n" + "configs=" + config_to_append + "\n"

    #TODO should we write this file
    f = open("score.py", "w+")
    f.write(source_with_config)
    f.close()


def deploy(task_name,
           model_name,
           deployment_type,
           local_model_path=None,
           aml_model_tag=None,
           aml_deployment_name=None,
           aml_workspace=None,
           aks_target=None,
           aks_deploy_config=None,
           parallelism_config={}):

    check_if_supported(task_name, model_name)
    create_score_file(task_name, model_name, parallelism_config)
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
    mii.import_score_file().init()
