'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import torch
import subprocess
import os
import yaml
import string

import mii

from .constants import DeploymentType, MII_MODEL_PATH_DEFAULT
from .utils import logger
from .models.score import create_score_file


def deploy(task,
           model,
           deployment_name,
           deployment_type=DeploymentType.LOCAL,
           model_path=None,
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

        model_path: Optional: In LOCAL deployments this is the local path where model checkpoints are available. In AML deployments this
            is an optional relative path with AZURE_MODEL_DIR for the deployment.

        enable_deepspeed: Optional: Defaults to True. Use this flag to enable or disable DeepSpeed-Inference optimizations

        enable_zero: Optional: Defaults to False. Use this flag to enable or disable DeepSpeed-ZeRO inference

        ds_config: Optional: Defaults to None. Use this to specify the DeepSpeed configuration when enabling DeepSpeed-ZeRO inference

        force_register_model: Optional: Defaults to False. For AML deployments, set it to True if you want to re-register your model
            with the same ``aml_model_tags`` using checkpoints from ``model_path``.

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

    # aml only allows certain characters for deployment names
    if deployment_type == DeploymentType.AML:
        allowed_chars = set(string.ascii_lowercase + string.ascii_uppercase +
                            string.digits + '-')
        assert set(deployment_name) <= allowed_chars, "AML deployment names can only contain a-z, A-Z, 0-9, and '-'"

    task = mii.utils.get_task(task)
    mii.utils.check_if_task_and_model_is_valid(task, model)
    if enable_deepspeed:
        mii.utils.check_if_task_and_model_is_supported(task, model)

    logger.info(f"*************DeepSpeed Optimizations: {enable_deepspeed}*************")

    # In local deployments use default path if no model path set
    if model_path is None and deployment_type == DeploymentType.LOCAL:
        model_path = MII_MODEL_PATH_DEFAULT
    elif model_path is None and deployment_type == DeploymentType.AML:
        model_path = "model"

    create_score_file(deployment_name=deployment_name,
                      deployment_type=deployment_type,
                      task=task,
                      model_name=model,
                      ds_optimize=enable_deepspeed,
                      ds_zero=enable_zero,
                      ds_config=ds_config,
                      mii_config=mii_config,
                      model_path=model_path)

    if deployment_type == DeploymentType.AML:
        _deploy_aml(deployment_name, model_path, model)
    elif deployment_type == DeploymentType.LOCAL:
        return _deploy_local(deployment_name, model_path=model_path)
    else:
        raise Exception(f"Unknown deployment type: {deployment_type}")


def _deploy_local(deployment_name, model_path):
    mii.utils.import_score_file(deployment_name).init()


def _fill_template(template, replace_dict):
    for var, val in replace_dict.items():
        template = template.replace(var, val)
    return template


def _deploy_aml(deployment_name, model_path, model_name):
    # Test azure-cli login
    try:
        acr_name = subprocess.check_output(
            ["az",
             "ml",
             "workspace",
             "show",
             "--query",
             "container_registry"],
            text=True)
        acr_name = acr_name.strip().replace('"', '').rsplit('/', 1)[-1]
    except subprocess.CalledProcessError as e:
        print("\n", "-" * 30, "\n")
        print("Unable to obtain ACR name from Azure-CLI. Please verify that you:")
        print("\t- Have Azure-CLI installed")
        print("\t- Are logged in to an active account on Azure-CLI")
        print("\t- Have Azure-CLI ML plugin installed")
        print("\n", "-" * 30, "\n")
        raise (e)

    # Values
    #TODO Assert deployment name has only [A-Za-z0-9] (and "-")
    output_dir = mii.utils.mii_aml_output_path(deployment_name)
    code_path = os.path.join(output_dir, "code")
    model_path = os.path.join(output_dir, "model")
    version = "2"
    endpoint_name = deployment_name + "-endpoint"
    environment_name = deployment_name + "-environment"
    image_name = deployment_name + "-image"
    replace_dict = {
        "<deployment-name>": deployment_name,
        "<model-name>": model_name,
        "<acr-name>": acr_name,
        "<code-path>": code_path,
        "<model-path>": model_path,
        "<version>": version,
        "<endpoint-name>": endpoint_name,
        "<environment-name>": environment_name,
        "<image-name>": image_name,
    }

    # Make output dirs
    os.makedirs(code_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "build", "runit", "gunicorn"), exist_ok=True)

    # Fill yaml templates
    template_dict = {
        "deployment": mii.templates.deployment_template,
        "endpoint": mii.templates.endpoint_template,
        "environment": mii.templates.environment_template
    }
    for template_type, template in template_dict.items():
        output_file = os.path.join(output_dir, f"{template_type}.yml")
        yaml_data = _fill_template(template, replace_dict)
        with open(output_file, "w") as f:
            yaml.dump(yaml.safe_load(yaml_data), f)

    # Fill deploy.sh
    output_file = os.path.join(output_dir, "deploy.sh")
    script_data = _fill_template(mii.templates.deploy_template, replace_dict)
    with open(output_file, "w") as f:
        f.write(script_data)

    # Docker files
    output_file = os.path.join(output_dir, "build", "Dockerfile")
    with open(output_file, "w") as f:
        f.write(mii.templates.dockerfile)

    output_file = os.path.join(output_dir, "build", "gunicorn_app")
    with open(output_file, "w") as f:
        f.write(mii.templates.gunicorn)

    output_file = os.path.join(output_dir, "build", "runit", "gunicorn", "run")
    with open(output_file, "w") as f:
        f.write(mii.templates.gunicorn_run)

    output_file = os.path.join(output_dir, "build", "runit", "gunicorn", "finish")
    with open(output_file, "w") as f:
        f.write(mii.templates.gunicorn_finish)

    output_file = os.path.join(output_dir, "build", "requirements.txt")
    with open(output_file, "w") as f:
        f.write(mii.templates.requirements)
