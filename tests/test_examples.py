import pytest

import mii


def deploy_local(name: str, model_name: str, config_kwargs: dict):
    mii_configs = mii.constants.MII_CONFIGS_DEFAULT
    for key, val in config_kwargs:
        mii_configs[key] = val

    mii.deploy(
        name,
        model_name,
        mii.DeploymentType.LOCAL,
        deployment_name=name + "_deployment",
        local_model_path=".cache/models/" + name,
        mii_configs=mii_configs,
    )


def query_local(name: str, query: dict):
    generator = mii.mii_query_handle(name + "_deployment")
    result = generator.query(query)
    return result


@pytest.mark.paremetrize(
    "name, model_name, config_kwargs, query, expected",
    [
        (
            "conversational",
            "microsoft/DialoGPT-small",
            {},
            {
                "text": str,
                "conversation_id": 3,
                "past_user_inputs": [],
                "generated_responses": [],
            },
            "",
        ),
        (
            "fill-mask",
            "bert-base-uncased",
            {},
            {"query": "Hello I'm a " + mask + " model."},
            "",
        ),
        (
            "question-answering",
            "deepset/roberta-large-squad2",
            {
                mii.constants.TENSOR_PARALLEL_KEY: 1,
                mii.constants.PORT_NUMBER_KEY: 50050,
            },
            {
                "question": "What is the greatest?",
                "context": "DeepSpeed is the greatest",
            },
            "",
        ),
        (
            "text-classification",
            "roberta-large-mnli",
            {},
            {"query": "DeepSpeed is the greatest"},
            "",
        ),
        (
            "text-generation",
            "distilgpt2",
            {},
            {"query": "DeepSpeed is the greatest"},
            "",
        ),
        (
            "token-classification",
            "Jean-Baptiste/roberta-large-ner-english",
            {},
            {"query": "My name is jean-baptiste and I live in montreal."},
            "",
        ),
    ],
)
def test_example_local(
    name: str, model_name: str, config_kwargs: dict, query: dict, expected: str
):
    deploy_local(name, model_name, config_kwargs)
    result = query_local(name, query)
    assert result == expected
