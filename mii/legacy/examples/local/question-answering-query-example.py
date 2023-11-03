# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii

name = "deepset/roberta-large-squad2"

generator = mii.mii_query_handle(name + "-qa-deployment")
results = generator.query({
    'question': "What is the greatest?",
    'context': "DeepSpeed is the greatest"
})
print(results.response)
print(f"time_taken: {results.time_taken}")
