# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import pytest
import mii


def test_multi_deploy(deployment_tag, multi_deployment, multi_query):
    generator = mii.mii_query_handle(deployment_tag)
    for query in multi_query:
        result = generator.query(query)
    assert result


@pytest.mark.parametrize(
    "task_name, model_name, query",
    [
        (
            "text-generation",
            "bigscience/bloom-560m",
            {
                "query": ["DeepSpeed is the greatest",
                          'Seattle is']
            },
        ),
    ],
)
def test_partial_deploy(deployment_tag, multi_deployment, deployment_config, query):
    generator = mii.mii_query_handle(deployment_tag)
    generator.add_models(**deployment_config.__dict__)
    query["deployment_name"] = deployment_config.deployment_name
    result = generator.query(query)
    generator.delete_model(deployment_config.deployment_name)
    assert result
