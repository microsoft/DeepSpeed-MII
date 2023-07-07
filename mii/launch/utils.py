# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import base64
import json


def decode_config_from_str(config_str):
    # str -> bytes
    b64_bytes = config_str.encode()
    # decode b64 bytes -> json bytes
    config_bytes = base64.urlsafe_b64decode(b64_bytes)
    # convert json bytes -> str -> dict
    return json.loads(config_bytes.decode())
