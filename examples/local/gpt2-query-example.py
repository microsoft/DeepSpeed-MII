import os
import grpc

import mii

generator = mii.mii_query_handle("gpt2_deployment")
results = generator.query({'query': "DeepSpeed is the greatest"})
print(results)
