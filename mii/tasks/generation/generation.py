import asyncio
import torch
import sys
import os
import subprocess 
import time
import grpc
import mii


def generation_query_handle(port_number=50050):
    return MIIGenerationServerClient("na", "na", initialize_generator_service=False, initialize_grpc_client=True, use_grpc_server=True, port_number=port_number)


class MIIGenerationServerClient():
    '''Initialize the model, setup the server and client for the model under model_path'''
    def __init__(self, model_name, model_path, initialize_generator_service=True, initialize_grpc_client=True, use_grpc_server=False, port_number=50050):
        self.num_gpus = torch.cuda.device_count()
        assert self.num_gpus > 0, "No GPU detected"
        
        # This is true in two cases
        # i) If its multi-GPU
        # ii) It its a local deployment without AML
        self.use_grpc_server = True if (self.num_gpus > 1) else use_grpc_servver
        self.initialize_generator_service=initialize_generator_service
        self.initialize_grpc_client = initialize_grpc_client
        
        self.port_number = port_number

        if initialize_generator_service and not self.use_grpc_server:
            self.generator = None
                   
        if self.initialize_generator_service:
            self._initialize_generator_service(model_name, model_path)
        
        if self.initialize_grpc_client and self.use_grpc_server:   
            self.stubs = []
            self.asyncio_loop = asyncio.get_event_loop() 
            self._initialize_grpc_client() 

    def _initialize_generator_service(self, model_name, model_path):
       
        if not self.use_grpc_server:
            self.generator = mii.load_generator_models(model_name, model_path)
        else:
            #TODO path relative to the install
            # multi_gpu_server_script = os.path.join(os.environ.get('MII_PATH'),"initialize_multi_gpu_server.py")
            #TODO launch deepspeed from python path for Jeff
            #TODO we need to dump the log from these executions for debugging. Currently these are all lost
            ds_launch_str = f"deepspeed --num_gpus {self.num_gpus} --no_local_rank --no_python"
            py_bin = sys.executable
            launch_str = f"{py_bin} -m mii.launch.multi_gpu_server --model {model_name} --model-path {model_path} --port {self.port_number}"
            cmd = f'{ds_launch_str} {launch_str}'.split(" ")
            # cmd = ["deepspeed", "--num_gpus", f"{self.num_gpus}", multi_gpu_server_script , model_name, model_path, f"{self.port_number}"]
            print(cmd)
            process = subprocess.Popen(cmd)

            #TODO wait until the subprocess is ready. Currently just sleeping.. write a file hack 
            #TODO The above subprocess is ready might also be a good point untiil which we dump the stdout from the processes 
            #TODO maybe a heartbeat check       
            print("Done launching deepspeed, now waiting 15 seconds for the process to be fully initilaized")
            time.sleep(15)
            print("Done waiting")

    def _initialize_grpc_client(self):
        channels = []
        for i in range(self.num_gpus):
            channel = grpc.aio.insecure_channel(f'localhost:{self.port_number+ i}')
            stub = mii.grpc_related.proto.modelresponse_pb2_grpc.ModelResponseStub(channel)
            channels.append(channel)
            self.stubs.append(stub)

    #runs task in parallel and return the result from the first task
    async def _generate_in_tensor_parallel(self, request_string):
        responses = []
        for i in range(self.num_gpus):
            responses.append(self.asyncio_loop.create_task(self._request_response(i,request_string)))
        
        await responses[0]

        return responses[0]

    async def _request_response(self, stub_id, request):
        response = await self.stubs[stub_id].StringReply(mii.modelresponse_pb2.RequestString(request=request))
        return response

    def query(self, request_string):
        if not self.use_grpc_server:
            response = self.generator(request_string, do_sample=True, min_length=50)
            generated_string = f"{response}"
        else:
            assert self.initialize_grpc_client, "grpc client has not been setup when this generator was created"
            response = self.asyncio_loop.run_until_complete(self._generate_in_tensor_parallel(request_string))
            generated_string = response.result().response
        return generated_string
        
        