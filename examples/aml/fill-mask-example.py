# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii

name = "bert-base-uncased"
print(f"Deploying {name}...")

mii.deploy(task='fill-mask',
           model=name,
           deployment_name=name + "-deployment",
           deployment_type=mii.constants.DeploymentType.AML)
