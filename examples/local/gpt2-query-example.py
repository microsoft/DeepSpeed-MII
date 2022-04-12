import os
import grpc

import mii

name = "microsoft/DialoGPT-small"

generator = mii.mii_query_handle(name+"_deployment")
results = generator.query({'query': "DeepSpeed is the greatest"})
print(results)
