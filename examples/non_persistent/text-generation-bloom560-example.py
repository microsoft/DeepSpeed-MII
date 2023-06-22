# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii

mii_configs = {"tensor_parallel": 1, "dtype": "fp16"}
generator = mii.deploy_non_persistent(
    task='text-generation',
    model="bigscience/bloom-560m",
    deployment_name="bloom560m_deployment",
    deployment_type=mii.constants.DeploymentType.NON_PERSISTENT,
    mii_config=mii_configs)
result = generator.query({'query': ["DeepSpeed is the", "Seattle is"]})
print(result)
