import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import mii
generator = mii.generation_query_handle()
print(generator.query("DeepSpeed is the greatest"))