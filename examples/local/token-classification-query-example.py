# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii

# roberta
name = "Jean-Baptiste/roberta-large-ner-english"

print(f"Querying {name}...")

generator = mii.mii_query_handle(name + "_deployment")
result = generator.query({'query': "My name is jean-baptiste and I live in montreal."})
print(result.response)
print("time_taken:", result.time_taken)
