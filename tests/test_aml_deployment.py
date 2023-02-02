import pytest
import requests

import mii


@pytest.mark.local
@pytest.mark.parametrize("dtype", ['fp16', 'fp32'])
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
                          'Seattle is']
            },
        ),
        ("token-classification",
         "Jean-Baptiste/roberta-large-ner-english",
         {
             "query": "My name is jean-baptiste and I live in montreal."
         }),
        (
            "text-classification",
            "roberta-large-mnli",
            {
                "query": "DeepSpeed is the greatest"
            },
        ),
    ],
)
def test_single_GPU(aml_local_deployment, query):
    port = aml_local_deployment.mii_config["local_aml_port"]
    url = f"http://localhost:{port}/score"
    result = requests.post(url, json=query)
    assert json.load(result.json())["responses"]
