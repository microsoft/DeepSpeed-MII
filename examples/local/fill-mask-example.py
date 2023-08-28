# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import mii
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", action="store_true", help="query")
args = parser.parse_args()

name = "bert-base-uncased"
mask = "[MASK]"

if not args.query:
    print(f"Deploying {name}...")
    mii.deploy(task='fill-mask', model=name, deployment_name=name + "_deployment")
else:
    print(f"Querying {name}...")
    generator = mii.mii_query_handle(name + "_deployment")
    result = generator.query({'query': f"Hello I'm a {mask} model."})
    print(result.response)
    print("time_taken:", result.time_taken)
