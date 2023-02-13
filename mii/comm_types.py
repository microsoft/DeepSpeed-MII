from abc import ABC, abstractmethod

class BaseProtoComms(ABC):
    @abstractmethod
    def pack_request_to_proto(self, request_dict, **query_kwargs):
        ...

    @abstractmethod
    def unpack_request_from_proto(self, request):
        ...

    @abstractmethod
    def pack_response_to_proto(self, request_dict, **query_kwargs):
        ...

class 
