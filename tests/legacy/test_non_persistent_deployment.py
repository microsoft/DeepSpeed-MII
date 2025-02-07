# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import mii.legacy as mii


@pytest.mark.parametrize("deployment_type", [mii.DeploymentType.NON_PERSISTENT])
@pytest.mark.parametrize(
    "task_name, model_name, query",
    [
        (
            "fill-mask",
            "bert-base-uncased",
            {
                "query": "Hello I'm a [MASK] model."
            },
        ),
        (
            "question-answering",
            "deepset/roberta-large-squad2",
            {
                "question": "What is the greatest?",
                "context": "DeepSpeed is the greatest",
            },
        ),
        (
            "text-generation",
            "distilgpt2",
            {
                "query": ["DeepSpeed is the greatest"]
            },
        ),
        (
            "text-generation",
            "bigscience/bloom-560m",
            {
                "query": ["DeepSpeed is the greatest",
                          "Seattle is"]
            },
        ),
        (
            "token-classification",
            "Jean-Baptiste/roberta-large-ner-english",
            {
                "query": "My name is jean-baptiste and I live in montreal."
            },
        ),
        (
            "text-classification",
            "roberta-large-mnli",
            {
                "query": "DeepSpeed is the greatest"
            },
        ),
    ],
)
def test_single_GPU(deployment, query):
    generator = mii.mii_query_handle(deployment)
    result = generator.query(query)
    assert result
