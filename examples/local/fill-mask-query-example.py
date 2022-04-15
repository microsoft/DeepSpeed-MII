import os
import grpc

import mii

mii_configs = mii.constants.MII_CONFIGS_DEFAULT
mii_configs[mii.constants.TENSOR_PARALLEL_KEY] = 2

# roberta
name = "roberta-base"

print(f"Querying {name}...")

generator = mii.mii_query_handle(name + "_deployment")
result = generator.query({'query': "Hello I'm a <mask> model."})
print(result.response)
print("time_taken:", result.time_taken)
