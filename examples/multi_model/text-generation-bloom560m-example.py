# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii

deployments = []
mii_configs1 = {"tensor_parallel": 1, "dtype": "fp16"}
deployments.append(mii.Deployment(task='text-generation',
           model="bigscience/bloom-560m",
           deployment_name="bloom560m_deployment",
           mii_config=mii.config.MIIConfig(**mii_configs1)))

# gpt2
name = "microsoft/DialogRPT-human-vs-rand"
deployments.append(mii.Deployment(task='text-classification', model=name, deployment_name=name + "_deployment"))

mii.deploy(deployment_tag="first_test", deployments=deployments)
