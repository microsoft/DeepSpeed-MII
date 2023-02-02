import pytest

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
def test_single_GPU(local_deployment, query):
    generator = mii.mii_query_handle(local_deployment.deployment_name)
    result = generator.query(query)
    assert result


@pytest.mark.local
@pytest.mark.parametrize("load_with_sys_mem", [True])
@pytest.mark.parametrize(
    "task_name, model_name, query",
    [
        (
            "text-generation",
            "distilgpt2",
            {
                "query": ["DeepSpeed is the greatest"]
            },
        ),
    ],
)
def test_load_to_sys_mem(local_deployment, query):
    generator = mii.mii_query_handle(local_deployment.deployment_name)
    result = generator.query(query)
    assert result


@pytest.mark.local
@pytest.mark.parametrize(
    "task_name, model_name, query, enable_deepspeed, enable_zero, ds_config",
    [
        (
            "text-generation",
            "distilgpt2",
            {
                "query": "DeepSpeed is the greatest"
            },
            False,
            True,
            {
                "fp16": {
                    "enabled": False
                },
                "bf16": {
                    "enabled": False
                },
                "zero_optimization": {
                    "stage": 3,
                    "offload_param": {
                        "device": "cpu",
                    },
                },
                "train_micro_batch_size_per_gpu": 1,
            },
        ),
    ],
)
def test_zero_config(local_deployment, query):
    generator = mii.mii_query_handle(local_deployment.deployment_name)
    result = generator.query(query)
    assert result


@pytest.mark.local
@pytest.mark.parametrize("expected_failure", [AssertionError])
@pytest.mark.parametrize("enable_deepspeed, enable_zero, dtype",
                         [(True,
                           True,
                           'fp32'),
                          (False,
                           True,
                           'fp16')])
@pytest.mark.parametrize("ds_config",
                         [
                             {
                                 "fp16": {
                                     "enabled": False
                                 },
                                 "bf16": {
                                     "enabled": False
                                 },
                                 "zero_optimization": {
                                     "stage": 3,
                                     "offload_param": {
                                         "device": "cpu",
                                     },
                                 },
                                 "train_micro_batch_size_per_gpu": 1,
                             },
                         ])
@pytest.mark.parametrize(
    "task_name, model_name, query",
    [
        (
            "text-generation",
            "distilgpt2",
            {
                "query": "DeepSpeed is the greatest"
            },
        ),
    ],
)
def test_zero_config_fail(local_deployment, query):
    print(local_deployment)
    assert "MII Config Error" in str(local_deployment.value)


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
