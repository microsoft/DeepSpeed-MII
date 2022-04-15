import os
import grpc

import mii

mii_configs = mii.constants.MII_CONFIGS_DEFAULT
mii_configs[mii.constants.TENSOR_PARALLEL_KEY] = 2

# gpt2
name = "microsoft/DialogRPT-human-vs-rand"

# roberta
name = "roberta-large-mnli"

print(f"Querying {name}...")

generator = mii.mii_query_handle(name + "_deployment")
results = generator.query({'query': "DeepSpeed is the greatest"})
print(results.response)
