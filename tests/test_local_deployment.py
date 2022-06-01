import pytest
from types import SimpleNamespace

import mii


def validate_config(config):
    if (config.model_name in ['bert-base-uncased']) and (config.mii_config['dtype']
                                                         == 'fp16'):
        pytest.skip(f"Model f{config.model_name} not supported for FP16")


@pytest.fixture(scope="function")
def config(task_name: str, model_name: str, mii_config: dict, query: dict):
    test_config = SimpleNamespace(task_name=task_name,
                                  model_name=model_name,
                                  mii_config=mii_config,
                                  query=query)
    validate_config(test_config)
    return test_config


@pytest.fixture(scope="function")
def local_deployment(config):
    mii.deploy(
        task_name=config.task_name,
        model_name=config.model_name,
        deployment_type=mii.DeploymentType.LOCAL,
        deployment_name=config.model_name + "_deployment",
        local_model_path=".cache/models/" + config.model_name,
        mii_configs=config.mii_config,
    )
    yield config
    mii.terminate_local_server(config.model_name + "_deployment")


@pytest.mark.local
@pytest.mark.parametrize("mii_config", [{'dtype': 'fp16'}, {'dtype': 'fp32'}])
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
                "query": "DeepSpeed is the greatest"
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
def test_single_GPU_local_deployment(local_deployment):
    generator = mii.mii_query_handle(local_deployment.model_name + "_deployment")
    result = generator.query(local_deployment.query)
    assert result


''' Not working yet
@pytest.mark.local
@pytest.mark.parametrize(
        "task_name, model_name, config, query",
        [
            (
            "text-generation",
            "gpt-neox",
            {"tensor_parallel": 2, 'dtype': 'fp16'},
            {"query": "DeepSpeed is the greatest"},
            )
        ]
    )
def test_multi_GPU_local_deployment(task_name:str, model_name:str, config:dict, query:dict):
    result = deploy_query_local(task_name=task_name, model_name=model_name, config=config, query=query)
    assert result
'''
