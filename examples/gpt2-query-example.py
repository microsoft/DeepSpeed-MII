import os
import grpc

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import mii

generator = mii.generation_query_handle()
results = generator.query("DeepSpeed is the greatest")
print(results)
