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
engine, and return the callable pipeline. A simple 4-line example is provided below:

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

**Any of the fields in** :class:`ModelConfig <mii.config.ModelConfig>` **can be
passed as keyword arguments or in a** ``model_config`` **dictionary to the**
:func:`mii.pipeline` **API. Please see** :ref:`Model Configuration
<model_configuration>` **for more information.**

Generate Options
----------------

The text-generation of the callable :class:`MIIPipeline
<mii.batching.ragged_batching.MIIPipeline>` class can be modified with several
keyword arguments. A full list of the available options can be found in
:class:`GenerateParamsConfig <mii.config.GenerateParamsConfig>`.

The generate options affect only the prompt(s) passed in a given call to the
pipeline. For example, you can control per-prompt generation length:

.. code-block:: python

    response_long = pipeline(prompt, max_length=1024)
    response_short = pipeline(prompt, max_length=128)

.. _pipeline_model_parallelism:

Model Parallelism
-----------------

Our pipeline object supports splitting models across multiple GPUs using tensor
parallelism. You must use the ``deepspeed`` launcher to enable tennsor parallelism
with the non-persistent pipeline, where the number of devices is controlled by
the ``--num_gpus <int>`` option.

As an example, consider the following ``example.py`` python script:

.. code-block:: python

    # example.py
    import mii
    pipe = mii.pipeline("mistralai/Mistral-7B-v0.1")

To run this pipeline on a single GPU, use ``python`` or ``deepspeed --num_gpus 1``:

.. code-block:: console

    (.venv) $ python example.py

To enable tensor parallelism across 2 GPUs, use ``deepspeed --num_gpus 2``:

.. code-block:: console

    (.venv) $ deepspeed --num_gpus 2 example.py

Because the ``deepspeed`` launcher will run multiple processes of
``example.py``, anything in the script will be executed by each process. For
example, consider the following script:

.. code-block:: python

    # example.py
    import os
    import mii
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    pipe = mii.pipeline("mistralai/Mistral-7B-v0.1")
    response = pipe("DeepSpeed is", max_length=16)
    print(f"rank {local_rank} response: {response}")

By default, the response is returned to only the rank 0 process. When run
with ``deepspeed --num_gpus 2 example.py`` the following output is produced:

.. code-block:: console

    (.venv) $ deepspeed --num_gpus 2 example.py
    rank 0 response: [a library for parallelizing and accelerating PyTorch.]
    rank 1 response: []

This behavior can be changed by enabling ``all_rank_output`` when creating the
pipeline (i.e., ``pipe = mii.pipeline("mistralai/Mistral-7B-v0.1",
all_rank_output=True)``):

.. code-block:: console

    (.venv) $ deepspeed --num_gpus 2 example.py
    rank 0 response: [a library for parallelizing and accelerating PyTorch.]
    rank 1 response: [a library for parallelizing and accelerating PyTorch.]
