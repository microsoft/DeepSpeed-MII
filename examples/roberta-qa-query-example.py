import os
import grpc

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import mii

generator = mii.mii_query_handle("question-answering")
results = generator.query({'question':"What is the greatest?", 'context': "DeepSpeed is the greatest"})
print(results)
