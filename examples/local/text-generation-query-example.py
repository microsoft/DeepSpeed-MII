import os
import grpc

import mii

name = "distilgpt2"

generator = mii.mii_query_handle(name + "_deployment")
results = generator.query({'query': "DeepSpeed is the greatest"})
print(results.response)
