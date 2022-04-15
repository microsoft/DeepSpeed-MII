from email import generator
import os
import grpc

import mii

mii_configs = mii.constants.MII_CONFIGS_DEFAULT
mii_configs[mii.constants.TENSOR_PARALLEL_KEY] = 2

# gpt2
name = "microsoft/DialoGPT-small"

print(f"Querying {name}...")

str = "DeepSpeed is the greatest"

generator = mii.mii_query_handle(name + "_deployment")
conv = generator.query({
    'text': str,
    'conversation_id': 3,
    'past_user_inputs': [],
    'generated_responses': []
})

print(conv)

str = "How is DeepSpeed?"
conv = generator.query({
    'text': str,
    'conversation_id': conv.conversation_id,
    'past_user_inputs': conv.past_user_inputs,
    'generated_responses': conv.generated_responses
})

print(conv)
