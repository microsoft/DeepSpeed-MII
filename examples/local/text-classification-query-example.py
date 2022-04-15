import os
import grpc

import mii

# gpt2
name = "microsoft/DialogRPT-human-vs-rand"

# roberta
name = "roberta-large-mnli"

print(f"Querying {name}...")

generator = mii.mii_query_handle(name + "_deployment")
result = generator.query({'query': "DeepSpeed is the greatest"})
print(result.response)
print("time_taken:", result.time_taken)
