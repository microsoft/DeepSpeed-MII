'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import json

import mii
from mii import utils
from mii.constants import DeploymentType
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


def create_score_file(deployment_name, task, model_name, ds_optimize, mii_configs):
    config_dict = {}
    config_dict[mii.constants.TASK_NAME_KEY] = mii.get_task_name(task)
    config_dict[mii.constants.MODEL_NAME_KEY] = model_name
    config_dict[mii.constants.ENABLE_DEEPSPEED_KEY] = ds_optimize
    config_dict[mii.constants.MII_CONFIGS_KEY] = mii_configs

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


def _get_inference_config(aml_workspace, deployment_name):
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

    inference_config = InferenceConfig(
        environment=Environment.get(aml_workspace,
                                    name=ENV_NAME),
        entry_script=utils.generated_score_path(deployment_name))
    return inference_config


def deploy(task_name,
           model_name,
           deployment_type,
           deployment_name,
           local_model_path=None,
           aml_model_tags=None,
           aml_workspace=None,
           aks_target=None,
           aks_deploy_config=None,
           force_register_model=False,
           enable_deepspeed=True,
           mii_configs=mii.constants.MII_CONFIGS_DEFAULT):
    """Deploy a task using specified model. For usage examples see:

        mii/examples/local/gpt2-local-example.py
        mii/examples/azure-local/gpt2-azure-local-example.py
        mii/examples/azure-aks/gpt2-azure-aks-example.py


    Arguments:
        task_name: Name of the machine learning task to be deployed.Currently MII supports the following list of tasks
            ``['text-generation', 'question-answering']``

        model_name: Name of a supported model for the task. Models in MII are sourced from multiple open-source projects
            such as Huggingface Transformer, FairSeq, EluetherAI etc. For the list of supported models for each task, please
            see here [TODO].

        deployment_type: One of the ``enum mii.DeploymentTypes: [LOCAL, AML_LOCAL, AML_ON_AKS]``.
            *``LOCAL`` uses a grpc server to create a local deployment, and query the model must be done by creating a query handle using
              `mii.mii_query_handle` and posting queries using ``mii_request_handle.query`` API,
            *``AML_LOCAL`` creates a local deployment using Azure ML and provides a URI where HTTP queries can be posted. This requires an azure subscription.
            *``AML_ON_AKS`` deploys the model using Azure ML on an Azure AKS-cluster and provides a URI where HTTP queries can be posted.
              This requires an azure subscription and resources to create an AKS-cluster. For both of the AML deployments, if deploying for the first time, the
              specified model is first registered and uploaded to your Azure Workspace and will be reused in subsequent deployments.

        deployment_name: Name of the deployment. Used as an identifier for posting queries for ``LOCAL`` deployment.
            For ``AML_ON_AKS`` deployment, this will be the name of the endpoint.


        local_model_path: Optional: Local folder where the model checkpoints are available.
            This should be provided if you want to use your own checkpoint instead of the default open-source checkpoints for the supported models for `LOCAL` deployment.
            For AML deployments, AML will look for the most recent model registered in your workspace for a given ``model_name`` and ``aml_model_tags``,
            and ignore the ``local_model_path`` if a matching ``model_name`` and ``aml_model_tags`` already exist. If you want MII to use the ``local_model_path`` with
            AML deployments, either set a new ``aml_model_tags`` to register the model from ``local_model_path`` to your Azure workspace, or set ``force_register_model=True``,
            to re-register the the model with the  existing ``model_name`` and ``aml_model_tags`` using the checkpoints in ``local_model_path``.


        aml_model_tags: Optional: Only needed for AML deployments. Tags used when registering your model to Azure Workspace, when deploying for the first time.
            For later deployments, tags are used as filters to identify the model from your workspace for AML based deployment.
            For details see here: https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model.model?view=azure-ml-py

        aml_workspace: Optional: Azure Workspace ``azureml.core.Workspace`` for AML deployment.
            For details see here: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?tabs=python#create-a-workspace


        aks_target: Optional: ``azureml.core.compute.ComputeTarget`` for `AML_ON_AKS` deployment created using AKS-Compute.
            It specifies the AKS-cluster in your Azure Workspace on which the model is deployed.
            For more details see here: https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.compute.akscompute?view=azure-ml-py


        aks_deploy_config: Optional: ``azureml.core.webservice.aks.AksServiceDeploymentConfiguration`` for `AML_ON_AKS` deployment.
            It specifies deployment configurations such as number of replicas, number of GPUs per replica, amount of memory, etc.
            It is created using `deploy_configuration` method of AKSWebService Class: https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice.akswebservice?view=azure-ml-py

        enable_deepspeed: Optional: Defaults to True. Use this flag to enable or disable DeepSpeed-Inference optimizations

        force_register_model: Optional: Defaults to False. For AML deployments, set it to True if you want to re-register your model
            with the same ``aml_model_tags`` using checkpoints from ``local_model_path``.

        mii_configs: Optional: Dictionary specifying optimization and deployment configurations. Defaults to ``mii.constants.MII_CONFIGS_DEFAULT``.
            mii_config is future looking to support extensions in optimization strategies supported by DeepSpeed Inference as we extend mii.
            As of now, it can be used to set tensor-slicing degree using mii.constants.TENSOR_PARALLEL_KEY and port number for deployment using mii.constants.PORT_NUMBER_KEY.
    Returns:
        If deployment_type is `LOCAL`, returns just the name of the deployment that can be used to create a query handle using `mii.mii_query_handle(deployment_name)`
        If deployment_type is `AML_LOCAL` or `AML_ON_AKS`, returns a a Webservice object from `azureml.core.webservice` corresponding to the deployed webservice
        For more details see here: https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice(class)?view=azure-ml-py.

    """

    task = mii.get_task(task_name)
    mii.check_if_task_and_model_is_supported(task, model_name)
    mii.utils.validate_mii_configs(mii_configs)

    logger.info(f"*************DeepSpeed Optimizations: {enable_deepspeed}*************")

    create_score_file(deployment_name, task, model_name, enable_deepspeed, mii_configs)

    if deployment_type == DeploymentType.LOCAL:
        return _deploy_local(deployment_name, local_model_path=local_model_path)

    if not azureml_available:
        raise RuntimeError(
            "azureml is not available, please install via 'pip install mii[azure]' to get extra dependencies"
        )

    assert aml_workspace is not None, "Workspace cannot be none for AML deployments"

    #either return a previously registered model, or register a new model
    model = _get_aml_model(task_name,
                           model_name,
                           aml_model_tags,
                           aml_workspace,
                           local_model_path=local_model_path,
                           force_register=force_register_model)

    logger.info(f"Deploying model {model}")

    #return
    inference_config = _get_inference_config(aml_workspace, deployment_name)

    if deployment_type == DeploymentType.AML_LOCAL:
        return _deploy_aml_local(model,
                                 inference_config,
                                 aml_workspace,
                                 deployment_name,
                                 mii_configs=mii_configs)

    assert aks_target is not None and aks_deploy_config is not None and deployment_name is not None, "aks_target and aks_deployment_config must be provided for AML_ON_AKS deployment"

    return _deploy_aml_on_aks(model,
                              inference_config,
                              aml_workspace,
                              aks_target,
                              aks_deploy_config,
                              deployment_name,
                              mii_configs=mii_configs)


def _deploy_aml_on_aks(model,
                       inference_config,
                       aml_workspace,
                       aks_target,
                       aks_deploy_config,
                       aml_deployment_name,
                       mii_configs=None):

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
                      mii_configs=None):

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


def _deploy_local(deployment_name, local_model_path=None):
    mii.set_model_path(local_model_path)
    mii.import_score_file(deployment_name).init()
