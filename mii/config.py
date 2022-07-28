import torch
from pydantic import BaseModel, validator


class MIIConfig(BaseModel):
    tensor_parallel: int = 1
    port_number: int = 50050
    dtype: str = "float"
    enable_cuda_graph: bool = False

    @validator('dtype')
    def dtype_valid(cls, value):
        # parse dtype value to determine torch dtype
        MIIConfig._torch_dtype(value)
        return value.lower()

    @staticmethod
    def _torch_dtype(value):
        value = value.lower()
        if value == "float" or value == "fp32":
            dtype = torch.float
        elif value == "half" or value == "fp16":
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
