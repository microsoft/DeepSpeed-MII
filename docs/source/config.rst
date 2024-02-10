Configuration
=============

The config classes described here are used to customize :doc:`pipeline` and :doc:`deployment`.

.. _model_configuration:

Model Configuration
-------------------

The :class:`ModelConfig <mii.config.ModelConfig>` is used to stand up a
DeepSpeed inference engine and provides a large amount of control to users. This
class is automatically generated from user-provided arguments to
:func:`mii.pipeline` and :func:`mii.serve`. The fields can be provided in a
``model_config`` dictionary or as keyword arguments.

For example, to change the default ``max_length`` for token generation of a
pipeline, the following are equivalent:

As a keyword argument:

.. code-block:: python

    pipe = mii.pipeline("mistralai/Mistral-7B-v0.1", max_length=2048)

As a ``model_config`` dictionary:

.. code-block:: python

    pipe = mii.pipeline("mistralai/Mistral-7B-v0.1", model_config={"max_length": 2048})

.. autopydantic_model:: mii.config.ModelConfig

.. _mii_configuration:

MII Server Configuration
------------------------

The :class:`MIIConfig <mii.config.MIIConfig>` is used to stand up a
DeepSpeed-MII `gRPC <https://grpc.io/>`_ server and provide a large amount of
control to users. This class is automatically generated from user-provided
arguments to :func:`mii.serve`. The fields can be provided in a ``mii_config``
dictionary or as keyword arguments.

For example, to change the base port number used to to communicate with a
persistent deployment and the default ``max_length`` for token generation, the
following are equivalent:

As keyword arguments:

.. code-block:: python

    client = mii.serve("mistralai/Mistral-7B-v0.1", port_number=50055, max_length=2048)

As ``model_config`` and ``mii_config`` dictionaries:

.. code-block:: python

    client = mii.serve("mistralai/Mistral-7B-v0.1", mii_config={"port_number": 50055}, model_config={"max_length": 2048})

.. autopydantic_model:: mii.config.MIIConfig

Text-Generation Configuration
-----------------------------

The :class:`GenerateParamsConfig <mii.config.GenerateParamsConfig>` is used to
process user-provided keyword arguments passed to :class:`MIIPipeline
<mii.batching.ragged_batching.MIIPipeline>` and :class:`MIIClient
<mii.backend.client.MIIClient>` when doing text-generation.

.. autopydantic_model:: mii.config.GenerateParamsConfig
    :exclude-members: prompt_length
