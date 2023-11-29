Non-Persistent Pipelines
========================

A non-persistent pipeline can be created with the :func:`mii.pipeline` API. This
returns a non-persistent :class:`MIIPipeline
<mii.batching.ragged_batching.MIIPipeline>` object that is destroyed when the
python script exits.

MIIPipeline
-----------

.. autoclass::
    mii.batching.ragged_batching.MIIPipeline

    .. automethod:: __call__

:class:`MIIPipeline <mii.batching.ragged_batching.MIIPipeline>` is a callable
class that provides a simplified interface for generating text for prompt
inputs. To create a pipeline, you must only provide the HuggingFace model name
(or path to a locally stored model) to the :func:`mii.pipeline` API.
DeepSpeed-MII will automatically load the model weights, create an inference
engine, and return the callable pipeline. We provide a simple 4-line example below:

.. code-block:: python

    import mii
    pipe = mii.pipeline("mistralai/Mistral-7B-v0.1")
    response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
    print(response)

Pipeline Configuration
----------------------

While we prioritize offering a simple interface to load models and run
text-generation, we also provide many configuration options for users that want
to customize the pipeline.

**Any of the fields documented in** :class:`ModelConfig
<mii.config.ModelConfig>` **can be passed as keyword arguments or in a**
``model_config`` **dictionary to the** :func:`mii.pipeline` **API.**

For example, if we want to change the default `max_length` for token generation, the following are equivalent:

As a keyword argument:

.. code-block:: python

    pipe = mii.pipeline("mistralai/Mistral-7B-v0.1", max_length=2048)

As a `model_config` dictionary:

.. code-block:: python

    pipe = mii.pipeline("mistralai/Mistral-7B-v0.1", model_config={"max_length": 2048})


Generate Options
----------------
