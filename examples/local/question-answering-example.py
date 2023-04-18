# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii

mii_config = {'tensor_parallel': 1, 'port_number': 50050}

name = "deepset/roberta-large-squad2"
mii.deploy(task="question-answering",
           model=name,
           deployment_name=name + "-qa-deployment",
           mii_config=mii_config)
