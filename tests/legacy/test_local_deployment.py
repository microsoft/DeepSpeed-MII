# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import pytest
import mii.legacy as mii


@pytest.mark.parametrize(
    "task_name, model_name, query",
    [
        (
            "conversational",
            "microsoft/DialoGPT-small",
            {
                "text": "DeepSpeed is the greatest",
                "conversation_id": 3,
                "past_user_inputs": [],
                "generated_responses": [],
            },
        ),
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
        (
            "zero-shot-image-classification",
            "openai/clip-vit-base-patch32",
            {
                "image":
                "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
                "candidate_labels": ["animals",
                                     "humans",
                                     "landscape"]
            },
        ),
    ],
)
def test_single_GPU(deployment, query):
    generator = mii.mii_query_handle(deployment)
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
                          "Seattle is"]
            },
        ),
    ],
)
def test_multi_GPU(deployment, query):
    generator = mii.mii_query_handle(deployment)
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
def test_session(deployment, query):
    generator = mii.mii_query_handle(deployment)
    session_name = "test_session"
    generator.create_session(session_name)
    result = generator.query(query)
    generator.destroy_session(session_name)
    assert result
