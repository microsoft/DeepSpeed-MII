# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import subprocess
import yaml
import mii


def get_acr_name():
    try:
        acr_name = subprocess.check_output(
            ["az",
             "ml",
             "workspace",
             "show",
             "--query",
             "container_registry"],
            text=True)
        return acr_name.strip().replace('"', '').rsplit('/', 1)[-1]
    except subprocess.CalledProcessError as e:
        print("\n", "-" * 30, "\n")
        print("Unable to obtain ACR name from Azure-CLI. Please verify that you:")
        print(
            "\t- Have Azure-CLI installed (https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)"
        )
        print("\t- Are logged in to an active account on Azure-CLI ($az login)")
        print("\t- Have Azure-CLI ML plugin installed ($az extension add --name ml)")
        print("\t- You have the default subscription, resource group, and workspace set")
        print("\t\t- az account set --subscription YOUR_SUBSCRIPTION")
        print("\t\t- az config set defaults.group=YOUR_GROUP")
        print("\t\t- az config set defaults.workspace=YOUR_WORKSPACE")
        print("\n", "-" * 30, "\n")
        raise (e)


def aml_output_path(deployment_name):
    output_path = os.path.join(os.getcwd(), f"{deployment_name}_aml")
    os.makedirs(output_path, exist_ok=True)
    return output_path


def fill_template(template, replace_dict):
    for var, val in replace_dict.items():
        template = template.replace(var, val)
    return template


def write_out_script(output_file, script):
    dir_path = os.path.dirname(output_file)
    os.makedirs(dir_path, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(script)


def write_out_yaml(output_file, yaml_data):
    dir_path = os.path.dirname(output_file)
    os.makedirs(dir_path, exist_ok=True)
    with open(output_file, "w") as f:
        yaml.dump(yaml.safe_load(yaml_data), f)


def generate_aml_scripts(acr_name,
                         deployment_name,
                         model_name,
                         task_name,
                         replica_num,
                         instance_type,
                         version):
    output_dir = aml_output_path(deployment_name)
    code_path = os.path.join(output_dir, "code")
    model_path = os.path.join(output_dir, "model")
    endpoint_name = deployment_name + "-endpoint"
    environment_name = deployment_name + "-environment"
    image_name = deployment_name + "-image"

    # Dictionary to fill template values
    replace_dict = {
        "<acr-name>": acr_name,
        "<deployment-name>": deployment_name,
        "<model-name>": model_name,
        "<task-name>": task_name,
        "<replica-num>": str(replica_num),
        "<instance-type>": instance_type,
        "<version>": str(version),
        "<code-path>": code_path,
        "<model-path>": model_path,
        "<endpoint-name>": endpoint_name,
        "<environment-name>": environment_name,
        "<image-name>": image_name,
    }

    # Docker files
    write_out_script(os.path.join(output_dir,
                                  "build",
                                  "Dockerfile"),
                     fill_template(mii.aml_related.templates.dockerfile,
                                   replace_dict))
    write_out_script(os.path.join(output_dir,
                                  "build",
                                  "gunicorn_app"),
                     fill_template(mii.aml_related.templates.gunicorn,
                                   replace_dict))
    write_out_script(os.path.join(output_dir,
                                  "build",
                                  "runit",
                                  "gunicorn",
                                  "run"),
                     fill_template(mii.aml_related.templates.gunicorn_run,
                                   replace_dict))
    write_out_script(
        os.path.join(output_dir,
                     "build",
                     "runit",
                     "gunicorn",
                     "finish"),
        fill_template(mii.aml_related.templates.gunicorn_finish,
                      replace_dict))
    write_out_script(os.path.join(output_dir,
                                  "build",
                                  "requirements.txt"),
                     fill_template(mii.aml_related.templates.requirements,
                                   replace_dict))

    # Model download script
    write_out_script(
        os.path.join(output_dir,
                     "model_download.py"),
        fill_template(mii.aml_related.templates.model_download,
                      replace_dict))

    # Deployment script
    write_out_script(os.path.join(output_dir,
                                  "deploy.sh"),
                     fill_template(mii.aml_related.templates.deploy,
                                   replace_dict))

    # Yaml configs
    write_out_yaml(os.path.join(output_dir,
                                "deployment.yml"),
                   fill_template(mii.aml_related.templates.deployment,
                                 replace_dict))
    write_out_yaml(os.path.join(output_dir,
                                "endpoint.yml"),
                   fill_template(mii.aml_related.templates.endpoint,
                                 replace_dict))
    write_out_yaml(os.path.join(output_dir,
                                "environment.yml"),
                   fill_template(mii.aml_related.templates.environment,
                                 replace_dict))
