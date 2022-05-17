from pydantic import BaseModel

class MIIConfig(BaseModel):
    tensor_parallel: int = 1
    port_number: int = 50050

