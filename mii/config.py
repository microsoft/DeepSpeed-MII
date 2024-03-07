# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import string
from typing import List, Optional, Union, Dict, Any, Literal

from deepspeed.launcher.runner import DLTS_HOSTFILE, fetch_hostfile
from deepspeed.inference import RaggedInferenceEngineConfig

from mii.constants import DeploymentType, TaskType, ModelProvider
from mii.errors import DeploymentNotFoundError
from mii.modeling.tokenizers import MIITokenizerWrapper
from mii.pydantic_v1 import BaseModel, Field, root_validator, validator, Extra
from mii.utils import generate_deployment_name, get_default_task, import_score_file

DEVICE_MAP_DEFAULT = "auto"


class MIIConfigModel(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        use_enum_values = True
        allow_population_by_field_name = True
        extra = "forbid"
        arbitrary_types_allowed = True


class GenerateParamsConfig(MIIConfigModel):
    """
    Options for changing text-generation behavior.
    """

    prompt_length: int
    """ Length of the input prompt. Autopopulated when creating requests, any user-provided values will be ignored."""

    max_length: int = 1024
    """ Maximum length of ``input_tokens`` + ``generated_tokens``. """

    max_new_tokens: int = None
    """ Maximum number of new tokens generated. ``max_length`` takes precedent. """

    min_new_tokens: int = 0
    """ Minimum number of new tokens generated. """

    stream: bool = False
    """ Enable streaming output. """

    ignore_eos: bool = False
    """ Ignore EoS token and continue generating text until we reach ``max_length`` or ``max_new_tokens``. """

    return_full_text: bool = False
    """ Prepends the input prompt to the generated text. """

    do_sample: bool = True
    """ When ``False``, do greedy sampling. """

    top_p: float = Field(0.9, gt=0, le=1)
    """ Top P value. """

    top_k: Optional[int] = Field(None, gt=0)
    """ Top K value. """

    temperature: Optional[float] = Field(None, gt=0)
    """ Temperature value. """

    stop: List[str] = []
    """ List of strings to stop generation at."""
    @validator("stop", pre=True)
    def make_stop_string_list(cls, field_value: Union[str, List[str]]) -> List[str]:
        if isinstance(field_value, str):
            return [field_value]
        return field_value

    @validator("stop")
    def sort_stop_strings(cls, field_value: List[str]) -> List[str]:
        return sorted(field_value)

    @root_validator
    def check_prompt_length(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        prompt_length = values.get("prompt_length")
        max_length = values.get("max_length")
        assert max_length > prompt_length, f"max_length ({max_length}) must be greater than prompt_length ({prompt_length})"
        return values

    @root_validator
    def set_max_new_tokens(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        max_length = values.get("max_length")
        max_new_tokens = values.get("max_new_tokens")
        prompt_length = values.get("prompt_length")
        if max_new_tokens is None:
            values["max_new_tokens"] = max_length - prompt_length
        return values

    class Config:
        extra = Extra.forbid


class ReplicaConfig(MIIConfigModel):
    hostname: str = ""
    tensor_parallel_ports: List[int] = []
    torch_dist_port: int = None
    gpu_indices: List[int] = []
    zmq_port: int = None


class ModelConfig(MIIConfigModel):
    model_name_or_path: str
    """
    Model name or path of the model to HuggingFace model to be deployed.
    """

    tokenizer: Optional[Union[str, MIITokenizerWrapper]] = None
    """
    Tokenizer wrapped with `MIITokenizerWrapper`, name or path of the
    HuggingFace tokenizer to be used.
    """

    task: Optional[TaskType] = TaskType.TEXT_GENERATION
    """
    Name of the task to be performed by the model.
    """

    tensor_parallel: int = int(os.getenv("WORLD_SIZE", "1"))
    """
    Tensor parallelism to use for a model (i.e., how many GPUs to shard a model
    across). This defaults to the `WORLD_SIZE` environment variable, or a value
    of 1 if that variable is not set. This value is also propagated to the
    `inference_engine_config`.
    """

    quantization_mode: Optional[str] = None
    """
    The quantization mode in string format. The supported modes are as follows:
        - 'wf6af16', weight-only quantization with FP6 weight and FP16 activation.
    """

    inference_engine_config: RaggedInferenceEngineConfig = {}
    """
    DeepSpeed inference engine config. This is automatically generated, but you
    can provide a set of custom configs.
    """

    torch_dist_port: int = 29500
    """
    Torch distributed port to be used. This also serves as a base port when
    multiple replicas are deployed. For example, if there are 2 replicas, the
    first will use port 29500 and the second will use port 29600.
    """

    zmq_port_number: int = 25555
    """
    Port number to use for the ZMQ communication (for broadcasting requests and
    responses among all ranks in ragged batching).
    """

    replica_num: int = Field(1, gt=0)
    """
    Number of model replicas. Enables easy data parallelism.
    """

    replica_configs: List[ReplicaConfig] = []
    """
    Configuration details for each replica. This will be automatically
    generated, but you can provide a set of custom configs.
    """

    device_map: Union[Literal["auto"], Dict[str, List[List[int]]]] = DEVICE_MAP_DEFAULT
    """
    GPU indices a model is deployed on. Note that CUDA_VISIBLE_DEVICES does not
    work with DeepSpeed-MII.
    """

    max_length: Optional[int] = None
    """
    The maximum number of tokens DeepSpeed-Inference can work with, including
    the input and output tokens.
    """

    sync_debug: bool = False
    """
    Inserts additional synchronization points for debugging purposes.
    """

    profile_model_time: bool = False
    """
    Log performance information about model inference with very little overhead.
    """
    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.HUGGING_FACE

    @validator("device_map", pre=True)
    def make_device_map_dict(cls, v):
        if isinstance(v, int):
            return {"localhost": [[v]]}
        if isinstance(v, list) and isinstance(v[0], int):
            return {"localhost": [v]}
        if isinstance(v, list) and isinstance(v[0], list):
            return {"localhost": v}
        return v

    @root_validator
    def auto_fill_values(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values.get("tokenizer"):
            values["tokenizer"] = values.get("model_name_or_path")
        if not values.get("task"):
            values["task"] = get_default_task(values.get("model_name_or_path"))
        return values

    @root_validator
    def propagate_tp_size(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        tensor_parallel = values.get("tensor_parallel")
        values.get("inference_engine_config").tensor_parallel.tp_size = tensor_parallel
        return values

    @root_validator
    def propagate_quantization_mode(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        quantization_mode = values.get("quantization_mode")
        values.get(
            "inference_engine_config").quantization.quantization_mode = quantization_mode
        return values

    @root_validator
    def check_replica_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        num_replica_config = len(values.get("replica_configs"))
        if num_replica_config > 0:
            assert num_replica_config == values.get("replica_num"), "Number of replica configs must match replica_num"
        return values


class MIIConfig(MIIConfigModel):
    deployment_name: str = ""
    """
    Name of the deployment. Used as an identifier for obtaining a inference
    server client and posting queries. Automatically generated if it is not provided.
    """

    deployment_type: DeploymentType = DeploymentType.LOCAL
    """
    One of the `enum mii.DeploymentTypes:`
    * `LOCAL` uses a grpc server to create a local deployment.
    * `AML` will generate the assets necessary to deploy on AML resources.
    """

    model_config: ModelConfig
    """
    Configuration for the deployed model(s).
    """

    port_number: int = 50050
    """
    Port number to use for the load balancer process.
    """

    enable_restful_api: bool = False
    """
    Enables a RESTful API that can be queries with via http POST method.
    """

    restful_api_host: str = "localhost"
    """
    Hostname to use for the RESTful API.
    """

    restful_api_port: int = 51080
    """
    Port number to use for the RESTful API.
    """

    restful_processes: int = Field(32, ge=1)
    """
    Number of processes to use for the RESTful API.
    """

    hostfile: str = DLTS_HOSTFILE
    """
    DeepSpeed hostfile. Will be autogenerated if None is provided.
    """

    # TODO: Place AML-related configs in subconfig
    version: int = 1
    """
    Version number to pass to AML deployments.
    """

    instance_type: str = "Standard_NC12s_v3"
    """
    AML instance type to use when create AML deployment assets.
    """
    @root_validator(skip_on_failure=True)
    def AML_name_valid(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get("deployment_type") == DeploymentType.AML:
            allowed_chars = set(string.ascii_lowercase + string.ascii_uppercase +
                                string.digits + "-")
            assert (
                set(values.get("deployment_name")) <= allowed_chars
            ), "AML deployment names can only contain a-z, A-Z, 0-9, and '-'."
        return values

    @root_validator(skip_on_failure=True)
    def check_deployment_name(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        deployment_name = values.get("deployment_name")
        if not deployment_name:
            model_name_or_path = values.get("model_config").model_name_or_path
            deployment_name = generate_deployment_name(
                model_name_or_path=model_name_or_path)
            values["deployment_name"] = deployment_name
        return values

    def generate_replica_configs(self) -> None:
        if self.model_config.replica_configs:
            return
        torch_dist_port = self.model_config.torch_dist_port
        tensor_parallel = self.model_config.tensor_parallel
        replica_pool = _allocate_devices(self.hostfile,
                                         tensor_parallel,
                                         self.model_config.replica_num,
                                         self.model_config.device_map)
        replica_configs = []
        for i, (hostname, gpu_indices) in enumerate(replica_pool):
            # Reserver port for a LB proxy when replication is enabled
            port_offset = 1
            base_port = self.port_number + i * tensor_parallel + port_offset
            tensor_parallel_ports = list(range(base_port, base_port + tensor_parallel))
            replica_torch_dist_port = torch_dist_port + (100 * i)
            replica_configs.append(
                ReplicaConfig(
                    hostname=hostname,
                    tensor_parallel_ports=tensor_parallel_ports,
                    torch_dist_port=replica_torch_dist_port,
                    gpu_indices=gpu_indices,
                    zmq_port=self.model_config.zmq_port_number + i,
                ))

        self.model_config.replica_configs = replica_configs


def _allocate_devices(hostfile_path: str,
                      tensor_parallel: int,
                      replica_num: int,
                      device_map: Dict[str,
                                       List[List[int]]] = DEVICE_MAP_DEFAULT):
    resource_pool = fetch_hostfile(hostfile_path)
    assert (
        resource_pool is not None and len(resource_pool) > 0
    ), f"No hosts found in {hostfile_path}"

    # If no device map was provided, we generate one based on the resources we find in the hostfile
    if device_map == DEVICE_MAP_DEFAULT:
        device_map = {}
        filled_slots = 0
        for host, slots in resource_pool.items():
            slots_to_fill = min(slots // tensor_parallel, replica_num - filled_slots)
            filled_slots += slots_to_fill
            device_map[host] = [
                list(range(i * tensor_parallel,
                           (i + 1) * tensor_parallel)) for i in range(slots_to_fill)
            ]

    # Assert that we have the correct number of mappings
    device_map_slots = sum([len(slots_list) for slots_list in device_map.values()])
    if device_map_slots < replica_num:
        raise ValueError(
            f"Only able to place {device_map_slots} replicas, but {replica_num} replicas were requested."
        )
    if device_map_slots > replica_num:
        raise ValueError(
            f"Device map contains {device_map_slots} mappings, but only {replica_num} replicas were requested. There must be a 1:1 mapping."
        )

    replica_pool = []
    # Fill the available slots with replicas
    for host, slots_list in device_map.items():
        if host not in resource_pool:
            raise ValueError(f"Host {host} not found in hostfile")
        for slots in slots_list:
            if len(slots) != tensor_parallel:
                raise ValueError(
                    f"Number of devices must match tensor_parallel. Found {len(slots)} devices for host {host}, but tensor_parallel={tensor_parallel}"
                )
            replica_pool.append((host, slots))

    return replica_pool


def get_mii_config(model_or_deployment_name: str) -> MIIConfig:
    try:
        deployment_name = model_or_deployment_name
        mii_config = import_score_file(deployment_name, DeploymentType.LOCAL).mii_config
    except:
        try:
            deployment_name = generate_deployment_name(
                model_name_or_path=model_or_deployment_name)
            mii_config = import_score_file(deployment_name,
                                           DeploymentType.LOCAL).mii_config
        except:
            raise DeploymentNotFoundError(
                f"Could not find a deployment named {model_or_deployment_name} or {deployment_name}"
            )
    return MIIConfig(**mii_config)
