# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii

mii_configs = {'tensor_parallel': 1}

# This checkpoint is create using DeepSpeed-Chat
# https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md
name = "AdamG012/chat-opt-1.3b-rlhf-actor-deepspeed"

print(f"Deploying {name}...")

# Deploy as "text-generation" task
mii.deploy(task='text-generation', model=name, deployment_name="chat_example_deployment")
