# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import mii

deployments = []
results = []
name = 'bigscience/bloom-560m'
mii_configs1 = {"tensor_parallel": 1, "dtype": "fp16"}
deployments.append(
    mii.DeploymentConfig(task='text-generation',
                         model=name,
                         deployment_name=name + "_deployment5",
                         mii_configs=mii.config.MIIConfig(**mii_configs1)
                         ))

generator = mii.mii_query_handle("multi_models")
generator.add_models(deployments=deployments)

result = generator.query(
    {
        "query": ["DeepSpeed is",
                  "Seattle is"],
        "deployment_name": "bigscience/bloom-560m_deployment5"
    },
    do_sample=True,
    max_new_tokens=30,
)
print(result)
generator.delete_model("bigscience/bloom-560m_deployment5")
