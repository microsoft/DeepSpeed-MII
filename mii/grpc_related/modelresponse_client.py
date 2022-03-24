# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC modelresponse.Greeter client."""

from __future__ import print_function

import logging

import grpc
import modelresponse_pb2
import modelresponse_pb2_grpc
import time
import sys
import asyncio

async def get_response(stub, request):
    response = await stub.StringReply(modelresponse_pb2.RequestString(request=request))
    return response



async def run(request):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    
    channel1 = grpc.aio.insecure_channel('localhost:50050')
    stub1 = modelresponse_pb2_grpc.ModelResponseStub(channel1)
    channel2 = grpc.aio.insecure_channel('localhost:50051')
    stub2 = modelresponse_pb2_grpc.ModelResponseStub(channel2)

    
    response1 = asyncio.create_task(get_response(stub1,request))
    print(f"First Request: {time.ctime()}")

    # with grpc.insecure_channel('localhost:500051') as channel:
    #     stub = modelresponse_pb2_grpc.ModelResponseStub(channel)
    #     response1 = stub.StringReply(modelresponse_pb2.RequestString(request=request))
        
    response2 = asyncio.create_task(get_response(stub2,request))
    print(f"Second Request: {time.ctime()}")
    
    await response1
    await response2

    # with grpc.insecure_channel('localhost:500052') as channel:
    #     stub = modelresponse_pb2_grpc.ModelResponseStub(channel)
    #     response2 = stub.StringReply(modelresponse_pb2.RequestString(request=request))
        
    print("Greeter client received: " + response1.result().response)
    print("Greeter client received: " + response2.result().response)
    print(f"Result : {time.ctime()}")
        

if __name__ == '__main__':
    logging.basicConfig()
    asyncio.run(run(sys.argv[1]))
