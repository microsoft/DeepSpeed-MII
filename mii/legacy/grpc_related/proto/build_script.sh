#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
python3 -m grpc_tools.protoc -I./ --python_out=. --grpc_python_out=. ./legacymodelresponse.proto

# update import to be global wrt mii
sed -i 's/legacymodelresponse_pb2/mii.legacy.grpc_related.proto.legacymodelresponse_pb2/g' legacymodelresponse_pb2_grpc.py
