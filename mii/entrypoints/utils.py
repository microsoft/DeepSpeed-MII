# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import uuid


def generate_short_uuid():
    return str(uuid.uuid4().hex)
