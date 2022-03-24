from load_models import load_generator_models
import sys, os
from modelresponse_server import serve

local_rank = int(os.getenv('LOCAL_RANK', '0'))
print(local_rank)
model_name = sys.argv[2]
model_path = sys.argv[3]
port = int(sys.argv[4]) + local_rank
generator = load_generator_models(model_name, model_path)
print(generator("Test product is ",  do_sample=True, min_length=50))
serve(generator, port)
