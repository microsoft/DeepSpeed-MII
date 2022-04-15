import os
import grpc

import mii

mii_configs = mii.constants.MII_CONFIGS_DEFAULT
mii_configs[mii.constants.TENSOR_PARALLEL_KEY] = 2

# roberta
name = "Jean-Baptiste/roberta-large-ner-english"

print(f"Querying {name}...")

generator = mii.mii_query_handle(name + "_deployment")
results = generator.query({'query': "My name is jean-baptiste and I live in montreal."})
print(results.response)
