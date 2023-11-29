Persistent Deployments
======================

A persistent model deployment can created with the :func:`mii.serve` API. This
stands up a gRPC server and returns a :class:`MIIClient
<mii.backend.client.MIIClient>` object that can be used to send generation
requests to the inference server. The inference server will persist after the
python script exits and until it is explicitly terminated.

MIIClient
---------

.. autoclass::
    mii.backend.client.MIIClient

    .. automethod:: __call__

    .. automethod:: generate

    .. automethod:: terminate_server
