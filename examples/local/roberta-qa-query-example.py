import os
import grpc

import mii

generator = mii.mii_query_handle("roberta-qa-deployment")
results = generator.query({
    'question': "What is the greatest?",
    'context': "DeepSpeed is the greatest"
})
print(results)
