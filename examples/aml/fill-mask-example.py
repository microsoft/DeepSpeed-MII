# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii

# roberta
name = "roberta-base"

name = "bert-base-uncased"

print(f"Deploying {name}...")

mii.deploy(task='fill-mask',
           model=name,
           deployment_name=name + "_deployment",
           deployment_type=mii.constants.DeploymentType.AML)
