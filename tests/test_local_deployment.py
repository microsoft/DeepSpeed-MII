import pytest

import mii


def deploy_local(task_name: str, model_name: str, config: dict):
    mii.deploy(
        task_name=task_name,
        model_name=model_name,
        deployment_type=mii.DeploymentType.LOCAL,
        deployment_name=model_name + "_deployment",
        local_model_path=".cache/models/" + model_name,
        mii_configs=config,
    )


def query_local(model_name: str, query: dict):
    generator = mii.mii_query_handle(model_name + "_deployment")
    result = generator.query(query)
    return result


def deploy_query_local(task_name: str, model_name: str, config: dict, query: dict):
    deploy_local(task_name, model_name, config)
    result = query_local(model_name, query)
    mii.terminate_local_server(model_name + "_deployment")
    return result


''' This one crashes. Need to debug later
        (
            "text-classification",
            "roberta-large-mnli",
            {"query": "DeepSpeed is the greatest"},
        ),
'''


@pytest.mark.local
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
    ],
)
def test_single_GPU_local_deployment(task_name: str, model_name: str, query: dict):
    config = {'tensor_parallel': 1}
    result = deploy_query_local(task_name=task_name,
                                model_name=model_name,
                                config=config,
                                query=query)
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
