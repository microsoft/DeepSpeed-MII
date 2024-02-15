Persistent Deployments
======================

A persistent model deployment can created with the :func:`mii.serve` API. This
stands up a gRPC server and returns a :class:`MIIClient
<mii.backend.client.MIIClient>` object that can be used to send generation
requests to the inference server. The inference server will persist after the
python script exits and until it is explicitly terminated.

To connect to an existing deployment, the :func:`mii.client` API is used. This
will connect with an existing gRPC server and return a :class:`MIIClient
<mii.backend.client.MIIClient>` object.

MIIClient
---------

.. autoclass::
    mii.backend.client.MIIClient

    .. automethod:: __call__

    .. automethod:: generate

    .. automethod:: terminate_server

:class:`MIIClient <mii.backend.client.MIIClient>` is a callable class that
provides a simplified interface for generating text for prompt inputs on a
persistent model deployment. To create a persistent deployment, you must only
provide the HuggingFace model name (or path to a locally stored model) to the
:func:`mii.serve` API. DeepSpeed-MII will automatically load the model weights,
create an inference engine, stand up a gRPC server, and return the callable
client. An example is provided below:

.. code-block:: python

    import mii
    client = mii.serve("mistralai/Mistral-7B-v0.1")
    response = client(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
    print(response)

Because the deployment is persistent, this server will continue running until it
is explicitly shutdown. This allows users to connect to a deployment from other
processes using the :func:`mii.client` API:

.. code-block:: python

    import mii
    client = mii.client("mistralai/Mistral-7B-v0.1")
    response = client(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
    print(response)

When a server needs to be shutdown, this can be done from any client object:

.. code-block:: python

    import mii
    client = mii.client("mistralai/Mistral-7B-v0.1")
    client.terminate_server()

Deployment Configuration
------------------------

While we prioritize offering a simple interface for loading models into
production-ready persistent deployments, we also provide many configuration
options for our persistent deployment.

**Any of the fields in** :class:`ModelConfig <mii.config.ModelConfig>` **and**
:class:`MIIConfig <mii.config.MIIConfig>` **can be passed as keyword
arguments or in respective** ``model_config`` **and** ``mii_config``
**dictionaries to the** :func:`mii.serve` **API. Please see** :ref:`Model
Configuration <model_configuration>` **and** :ref:`MII Server Configuration
<mii_configuration>` **for more information.**


Generate Options
----------------

Text-generation behavior using the callable :class:`MIIClient
<mii.backend.client.MIIClient>` class can be customized with several keyword
arguments. A full list of the available options can be found in
:class:`GenerateParamsConfig <mii.config.GenerateParamsConfig>`.

The generate options affect on the prompt(s) passed in a given call the client.
For example, the generation length can be controlled on a per-prompt basis and
override the default ``max_length``:

.. code-block:: python

    response_long = client(prompt, max_length=1024)
    response_short = client(prompt, max_length=128)

.. _deployment_model_parallelism:

Model Parallelism
-----------------

Our persistent deployment supports splitting models across multiple GPUs using
tensor parallelism. To enable model parallelism, pass the ``tensor_parallel``
argument to :func:`mii.serve`:

.. code-block:: python

    client = mii.serve("mistralai/Mistral-7B-v0.1", tensor_parallel=2)

.. _deployment_model_replicas:

Model Replicas
--------------

The persistent deployment can also create multiple model replicas. Passing the
``replica_num`` argument to :func:`mii.serve` enables this feature:

.. code-block:: python

    client = mii.serve("mistralai/Mistral-7B-v0.1", replica_num=2)

With multiple model replicas, the incoming requests from clients will be
forwarded to the replicas in a round-robin scheduling by an intermediate
load-balancer process. For example, if 4 requests with ids ``0, 1, 2, 3`` are
sent to the persistent deployment, then ``replica 0`` will process requests
``0`` and ``2`` while ``replica 1`` will process requests ``1`` and ``3``.

Model replicas also compose with model parallelism. For example, 2 replicas can
be created each split across 2 GPUs on a system with 4 GPUs total:

.. code-block:: python

    client = mii.serve("mistralai/Mistral-7B-v0.1", replica_num=2, tensor_parallel=2)
