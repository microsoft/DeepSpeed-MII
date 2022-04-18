import os
import grpc

import mii

name = "deepset/roberta-large-squad2"

generator = mii.mii_query_handle(name + "_deployment")
results = generator.query({
    'question': "What is the greatest?",
    'context': "DeepSpeed is the greatest"
})
print(results.response)
print(f"time_taken: {results.time_taken}")
