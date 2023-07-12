# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import mii

results = []
generator = mii.mii_query_handle("multi_models")
result = generator.query(
    {"query": ["DeepSpeed is",
               "Seattle is"]},
    "bloom560m_deployment",
    do_sample=True,
    max_new_tokens=30,
)
results.append(result)

result = generator.query({'query': "DeepSpeed is the greatest"},
                         "microsoft/DialogRPT-human-vs-rand_deployment")
results.append(result)

result = generator.query(
    {
        'text': "DeepSpeed is the greatest",
        'conversation_id': 3,
        'past_user_inputs': [],
        'generated_responses': []
    },
    "microsoft/DialoGPT-large_deployment")
results.append(result)

result = generator.query(
    {
        'question': "What is the greatest?",
        'context': "DeepSpeed is the greatest"
    },
    "deepset/roberta-large-squad2" + "-qa-deployment")
results.append(result)
