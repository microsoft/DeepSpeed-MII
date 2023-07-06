# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import mii
import time

generator = mii.mii_query_handle("first_test")
result = generator.query(
    {"query": ["DeepSpeed is",
               "Seattle is"]},
    "bloom560m_deployment",
    do_sample=True,
    max_new_tokens=30,
)
print(result)

time.sleep(5)
result = generator.query({'query': "DeepSpeed is the greatest"},
                         "microsoft/DialogRPT-human-vs-rand_deployment")
print(result)

time.sleep(5)

result = generator.query(
    {
        'text': "DeepSpeed is the greatest",
        'conversation_id': 3,
        'past_user_inputs': [],
        'generated_responses': []
    },
    "microsoft/DialoGPT-large_deployment")
print(result)
