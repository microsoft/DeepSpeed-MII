# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii

mii_configs = {"tensor_parallel": 1, "dtype": "fp16"}
name = "bloom560m"
mii.deploy(task='text-generation',
           model="bigscience/bloom-560m",
           deployment_name=name + "_deployment",
           deployment_type=mii.constants.DeploymentType.NON_PERSISTENT,
           mii_config=mii_configs)
generator = mii.mii_query_handle(name + "_deployment")
result = generator.query({'query': ["DeepSpeed is the", "Seattle is"]})
print(result)
