'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import asyncio
import grpc
import mii
from mii.utils import get_num_gpus
from mii.grpc_related.proto import modelresponse_pb2_grpc
from mii.constants import GRPC_MAX_MSG_SIZE
from mii.method_table import GRPC_METHOD_TABLE


def mii_query_handle(deployment_name):
    """Get a query handle for a local deployment:

        mii/examples/local/gpt2-query-example.py
        mii/examples/local/roberta-qa-query-example.py


    Arguments:
        deployment_name: Name of the deployment. Used as an identifier for posting queries for ``LOCAL`` deployment.

    Returns:
        query_handle: A query handle with a single method `.query(request_dictionary)` using which queries can be sent to the model.

    """

    configs = mii.utils.import_score_file(deployment_name).configs

    task = configs[mii.constants.TASK_NAME_KEY]

    assert task is not None, "The task name should be set before calling init"

    return mii.MIIClient(task, mii_configs=configs[mii.constants.MII_CONFIGS_KEY])


class MIIClient():
    '''Setup the client for the model'''
    def __init__(self, task_name, mii_configs={}):

        mii_configs = mii.config.MIIConfig(**mii_configs)

        self.task = mii.utils.get_task(task_name)

        self.num_gpus = get_num_gpus(mii_configs)
        assert self.num_gpus > 0, "GPU count must be greater than 0"

        self.port_number = mii_configs.port_number

        self.stubs = []
        self.asyncio_loop = asyncio.get_event_loop()
        self._initialize_grpc_client()

    def _initialize_grpc_client(self):
        channels = []
        for i in range(self.num_gpus):
            channel = grpc.aio.insecure_channel(f'localhost:{self.port_number + i}',
                                                options=[
                                                    ('grpc.max_send_message_length',
                                                     GRPC_MAX_MSG_SIZE),
                                                    ('grpc.max_receive_message_length',
                                                     GRPC_MAX_MSG_SIZE)
                                                ])
            stub = modelresponse_pb2_grpc.ModelResponseStub(channel)
            channels.append(channel)
            self.stubs.append(stub)

    # runs task in parallel and return the result from the first task
    async def _query_in_tensor_parallel(self, request_string, query_kwargs):
        responses = []
        for i in range(self.num_gpus):
            responses.append(
                self.asyncio_loop.create_task(
                    self._request_async_response(i,
                                                 request_string,
                                                 query_kwargs)))

        await responses[0]

        return responses[0]

    async def _request_async_response(self, stub_id, request_dict, query_kwargs):
        if self.task not in GRPC_METHOD_TABLE:
            raise ValueError(f"unknown task: {self.task}")

        conversions = GRPC_METHOD_TABLE[self.task]
        proto_request = conversions["pack_request_to_proto"](request_dict,
                                                             **query_kwargs)
        proto_response = await getattr(self.stubs[stub_id],
                                       conversions["method"])(proto_request)
        return conversions["unpack_response_from_proto"](
            proto_response
        ) if "unpack_response_from_proto" in conversions else proto_response

    def query(self, request_dict, **query_kwargs):
        """Query a local deployment:

            mii/examples/local/gpt2-query-example.py
            mii/examples/local/roberta-qa-query-example.py

        Arguments:
            request_dict: A task specific request dictionary consistinging of the inputs to the models
            query_kwargs: additional query parameters for the model

        Returns:
            response: Response of the model
        """
        response = self.asyncio_loop.run_until_complete(
            self._query_in_tensor_parallel(request_dict,
                                           query_kwargs))
        ret = response.result()

        return ret
