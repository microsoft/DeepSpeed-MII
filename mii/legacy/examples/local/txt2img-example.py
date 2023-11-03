# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import mii
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", action="store_true", help="query")
args = parser.parse_args()

if not args.query:
    mii_configs = {
        "tensor_parallel":
        1,
        "enable_cuda_graph":
        True,
        "replace_with_kernel_inject":
        True,
        "dtype":
        "fp16",
        "hf_auth_token":
        os.environ.get("HF_AUTH_TOKEN",
                       "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
        "port_number":
        50050
    }
    mii.deploy(task='text-to-image',
               model="runwayml/stable-diffusion-v1-5",
               deployment_name="sd_deploy",
               mii_config=mii_configs)
    print(
        "\nText to image model deployment complete! To use this deployment, run the following command: python txt2img-example.py --query\n"
    )
else:
    generator = mii.mii_query_handle("sd_deploy")
    result = generator.query({
        'query':
        ["a panda in space with a rainbow",
         "a soda can on top a snowy mountain"]
    })
    for idx, img in enumerate(result.images):
        img.save(f"test-{idx}.png")
