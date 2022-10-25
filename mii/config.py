import torch
from typing import Union, List
from pydantic import BaseModel, validator


class MIIConfig(BaseModel):
    tensor_parallel: int = 1
    port_number: int = 50050
    dtype: str = "float"
    enable_cuda_graph: bool = False
    checkpoint_dict: Union[dict, None] = None
    deploy_rank: Union[int, List[int]] = -1
    torch_dist_port: int = 29500
    hf_auth_token: str = None
    replace_with_kernel_inject: bool = True
    profile_model_time: bool = False

    @validator('dtype')
    def dtype_valid(cls, value):
        # parse dtype value to determine torch dtype
        MIIConfig._torch_dtype(value)
        return value.lower()

    @validator("deploy_rank")
    def deploy_valid(cls, field_value, values):
        if "tensor_parallel" not in values:
            raise ValueError(
                "'tensor_parallel' must be defined in the pydantic model before 'deploy_rank'"
            )

        # if deploy rank is not given, default to align with TP value
        if field_value == -1:
            field_value = list(range(values["tensor_parallel"]))

        # ensure deploy rank type is always list for easier consumption later
        if not isinstance(field_value, list):
            field_value = [field_value]

        # number of ranks provided must be equal to TP size, DP is handled outside MII currently
        assert values["tensor_parallel"] == len(field_value), \
            f"{len(field_value)} rank(s) provided in 'deploy_rank' does not align with tensor_parallel size of {values['tensor_parallel']}"
        return field_value

    @validator('checkpoint_dict')
    def checkpoint_dict_valid(cls, value):
        if value is None:
            return value
        if value.get('base_dir', ''):
            raise ValueError(
                "please unset 'base_dir' it will be set w.r.t. the deployment 'model_path'"
            )
        for k in ['checkpoints', 'parallelization', 'version', 'type']:
            if not value.get(k, ''):
                raise ValueError(f"Missing key={k} in checkpoint_dict")
        return value

    @staticmethod
    def _torch_dtype(value):
        value = value.lower()
        if value == "float" or value == "fp32" or value == "float32":
            dtype = torch.float
        elif value == "half" or value == "fp16" or value == "float16":
            dtype = torch.half
        elif value == "int8":
            dtype = torch.int8
        else:
            raise ValueError(f"unknown dtype={value}")
        return dtype

    def torch_dtype(self):
        return self._torch_dtype(self.dtype)

    class Config:
        validate_all = True
        validate_assignment = True
        use_enum_values = True
        extra = 'forbid'
