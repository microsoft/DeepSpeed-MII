import pytest
import functools
from types import SimpleNamespace

import mii


def validate_config(config):
    if (config.model_name in ['bert-base-uncased']) and (config.mii_config['dtype']
                                                         == 'fp16'):
        pytest.skip(f"Model f{config.model_name} not supported for FP16")


''' These fixtures provide default values for the deployment config '''


@pytest.fixture(scope="function", params=['fp32'])
def dtype(request):
    return request.param


@pytest.fixture(scope="function", params=[1])
def tensor_parallel(request):
    return request.param


@pytest.fixture(scope="function", params=[50050])
def port_number(request):
    return request.param


@pytest.fixture(scope="function", params=[True])
def enable_deepspeed(request):
    return request.param


@pytest.fixture(scope="function", params=[False])
def enable_zero(request):
    return request.param


@pytest.fixture(scope="function", params=[{}])
def ds_config(request):
    return request.param


''' These fixtures provide a local deployment and ensure teardown '''


@pytest.fixture(scope="function")
def mii_config(dtype: str, tensor_parallel: int, port_number: int):
    return {
        'dtype': dtype,
        'tensor_parallel': tensor_parallel,
        'port_number': port_number
    }


@pytest.fixture(scope="function")
def deployment_config(task_name: str,
                      model_name: str,
                      query: dict,
                      mii_config: dict,
                      enable_deepspeed: bool,
                      enable_zero: bool,
                      ds_config: dict):
    config = SimpleNamespace(task_name=task_name,
                             model_name=model_name,
                             query=query,
                             mii_config=mii_config,
                             enable_deepspeed=enable_deepspeed,
                             enable_zero=enable_zero,
                             ds_config=ds_config)
    validate_config(config)
    return config


@pytest.fixture(scope="function", params=[None])
def expected_failure(request):
    return request.param


@pytest.fixture(scope="function")
def local_deployment(deployment_config, expected_failure):
    if expected_failure is not None:
        with pytest.raises(expected_failure):
            mii.deploy(task_name=deployment_config.task_name,
                       model_name=deployment_config.model_name,
                       deployment_type=mii.DeploymentType.LOCAL,
                       deployment_name=deployment_config.model_name + "_deployment",
                       local_model_path=".cache/models/" + deployment_config.model_name,
                       mii_configs=deployment_config.mii_config,
                       enable_deepspeed=deployment_config.enable_deepspeed,
                       enable_zero=deployment_config.enable_zero,
                       ds_config=deployment_config.ds_config)
        yield None
    else:
        mii.deploy(task_name=deployment_config.task_name,
                   model_name=deployment_config.model_name,
                   deployment_type=mii.DeploymentType.LOCAL,
                   deployment_name=deployment_config.model_name + "_deployment",
                   local_model_path=".cache/models/" + deployment_config.model_name,
                   mii_configs=deployment_config.mii_config,
                   enable_deepspeed=deployment_config.enable_deepspeed,
                   enable_zero=deployment_config.enable_zero,
                   ds_config=deployment_config.ds_config)
        yield config
        mii.terminate_local_server(deployment_config.model_name + "_deployment")


@pytest.mark.local
@pytest.mark.parametrize("mii_config",
                         [{
                             'dtype': 'fp16',
                             'tensor_parallel': 1
                         },
                          {
                              'dtype': 'fp32',
                              'tensor_parallel': 1
                          }])
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
def test_single_GPU(local_deployment):
    generator = mii.mii_query_handle(local_deployment.model_name + "_deployment")
    result = generator.query(local_deployment.query)
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
def test_zero_config(local_deployment):
    generator = mii.mii_query_handle(local_deployment.model_name + "_deployment")
    result = generator.query(local_deployment.query)
    assert result


@pytest.mark.local
@pytest.mark.parametrize("enable_deepspeed, enable_zero, dtype",
                         [(True,
                           True,
                           'fp32'),
                          (False,
                           True,
                           'fp16')])
@pytest.mark.parametrize("expected_failure", [AssertionError])
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
def test_zero_config_fail(local_deployment):
    pass


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
