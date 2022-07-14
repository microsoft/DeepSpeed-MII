import os
import grpc

import mii

# name = "distilgpt2"
# name = "gpt2-xl"
#name = "facebook/opt-1.3b"
name = "bigscience/bloom"

generator = mii.mii_query_handle(name + "_deployment")
result = generator.query({'query': "DeepSpeed is the greatest"}, do_sample=False)
print(result.response)
print("time_taken:", result.time_taken)
