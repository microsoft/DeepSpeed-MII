API
===

DeepSpeed-MII provides a very simple API to deploy your LLM:

.. autofunction:: mii.pipeline

The :func:`mii.pipeline` API is a great way to try DeepSpeed-MII with ragged
batching and dynamic splitfuse. The pipeline is non-persistent and only exists
for the lifetime of the python script where it is used. For examples of how to
use :func:`mii.pipeline` please see :doc:`pipeline`.

.. autofunction:: mii.serve

The :func:`mii.serve` API is intended for production use cases, where a
persistent model deployment is necessary. The persistent deployment utilizes
ragged batching and dynamic splitfuse to deliver high throughput and low latency
to multiple clients in parallel. For examples of how to use :func:`mii.serve`
please see :doc:`deployment`.

.. autofunction:: mii.client

The :func:`mii.client` API allows multiple processes to connect to a persistent
deployment created with :func:`mii.serve`. For examples of how to use
:func:`mii.client` please see :doc:`deployment`.
