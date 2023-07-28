#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
python3 -m grpc_tools.protoc -I./ --python_out=. --grpc_python_out=. ./modelresponse.proto

# update import to be global wrt mii
sed -i 's/modelresponse_pb2/mii.grpc_related.proto.modelresponse_pb2/g' modelresponse_pb2_grpc.py
