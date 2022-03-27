import os
import grpc

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import mii

generator = mii.mii_query_handle("text-generation")
results = generator.query({'query':"DeepSpeed is the greatest"})
print(results)
