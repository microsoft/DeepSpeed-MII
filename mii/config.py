from pydantic import BaseModel


class MIIConfig(BaseModel):
    tensor_parallel: int = 1
    port_number: int = 50050

    class Config:
        validate_all = True
        validate_assignment = True
        use_enum_values = True
