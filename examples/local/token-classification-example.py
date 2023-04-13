# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii

# roberta
name = "Jean-Baptiste/roberta-large-ner-english"

print(f"Deploying {name}...")

mii.deploy(task='token-classification', model=name, deployment_name=name + "_deployment")
