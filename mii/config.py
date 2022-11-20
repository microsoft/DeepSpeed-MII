import torch
from typing import Union, List
from pydantic import BaseModel, validator
import socket


class MIIConfig(BaseModel):
    tensor_parallel: int = 1
    port_number: int = None
    dtype: str = "float"
    enable_cuda_graph: bool = False
    checkpoint_dict: Union[dict, None] = None
    deploy_rank: Union[int, List[int]] = -1
    torch_dist_port: int = 29500
    hf_auth_token: str = None
    replace_with_kernel_inject: bool = True
    profile_model_time: bool = False
    skip_model_check: bool = False

    def __is_port_in_use(port_number: int) -> bool:
        """
        Checks if a port_number is in use

        Args:
            port_number (int): port_number to check

        Returns:
            bool: True if port_number is in use else False
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port_number)) == 0

    @validator('port_number')
    def assign_port(port_number: int = None) -> int:
        """
        Starts a socket connection to grab a free port (Involves a race
            condition but will do for now)
        Args:
            port_number (int): Port to start the socket connection (default: None)
        Returns:
            int: Free port number
        """
        DEFAULT_PORT = 50050
        # if port is None set the default 50050 and default port is not in use return it
        if port_number is None:
            port_number = DEFAULT_PORT

        # if the defined port is in use find a free port
        if MIIConfig.__is_port_in_use(port_number):
            tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp.bind(("", 0))
            _, port_number = tcp.getsockname()
            tcp.close()

        MIIConfig.port_number = port_number

        return port_number

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
