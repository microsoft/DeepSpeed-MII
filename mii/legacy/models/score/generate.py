# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import mii.legacy as mii
import pprint
from mii.legacy.logging import logger
from mii.legacy.constants import DeploymentType


def create_score_file(mii_config):
    if len(mii.__path__) > 1:
        logger.warning(
            f"Detected mii path as multiple sources: {mii.__path__}, might cause unknown behavior"
        )

    with open(os.path.join(mii.__path__[0],
                           "models/score/score_template.py"),
              "r") as fd:
        score_src = fd.read()

    # update score file w. global config dict
    config_dict = mii_config.dict()
    source_with_config = f"{score_src}\n"
    source_with_config += f"mii_config = {pprint.pformat(config_dict, indent=4)}"

    with open(
            generated_score_path(mii_config.deployment_name,
                                 mii_config.deployment_type),
            "w") as fd:
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
