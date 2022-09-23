'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import os
import mii
import pprint
from mii.utils import logger
from mii.constants import DeploymentType


def create_score_file(deployment_name,
                      deployment_type,
                      task,
                      model_name,
                      ds_optimize,
                      ds_zero,
                      ds_config,
                      mii_config,
                      model_path):
    config_dict = {}
    config_dict[mii.constants.TASK_NAME_KEY] = mii.utils.get_task_name(task)
    config_dict[mii.constants.MODEL_NAME_KEY] = model_name
    config_dict[mii.constants.ENABLE_DEEPSPEED_KEY] = ds_optimize
    config_dict[mii.constants.MII_CONFIGS_KEY] = mii_config.dict()
    config_dict[mii.constants.ENABLE_DEEPSPEED_ZERO_KEY] = ds_zero
    config_dict[mii.constants.DEEPSPEED_CONFIG_KEY] = ds_config
    config_dict[mii.constants.MODEL_PATH_KEY] = model_path

    if len(mii.__path__) > 1:
        logger.warning(
            f"Detected mii path as multiple sources: {mii.__path__}, might cause unknown behavior"
        )

    with open(os.path.join(mii.__path__[0],
                           "models/score/score_template.py"),
              "r") as fd:
        score_src = fd.read()

    # update score file w. global config dict
    source_with_config = f"{score_src}\n"
    source_with_config += f"configs = {pprint.pformat(config_dict, indent=4)}"

    with open(generated_score_path(deployment_name, deployment_type), "w") as fd:
        fd.write(source_with_config)
        fd.write("\n")


def generated_score_path(deployment_name, deployment_type):
    if deployment_type == DeploymentType.LOCAL:
        score_path = os.path.join(mii.utils.mii_cache_path(), deployment_name)
    elif deployment_type == DeploymentType.AML:
        score_path = os.path.join(mii.aml_related.utils.aml_output_path(deployment_name),
                                  "code")
    if not os.path.isdir(score_path):
        os.makedirs(score_path)
    return os.path.join(score_path, "score.py")
