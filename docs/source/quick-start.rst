FastGen Quick Start Guide
=========================

This guide is aimed to get you quickly up and running DeepSpeed-MII and DeepSpeed-FastGen.

Requirements
------------

- 1 or more NVIDIA GPUs with >=sm_80 compute capability (e.g., A100, A6000)
- `PyTorch <https://pytorch.org/get-started/locally/>`_ installed in your local Python environment

Install
-------

Install the latest version of DeepSpeed-MII with the following:

.. code-block:: console

   (.venv) $ pip install -U deepspeed-mii

Run a Non-Persistent Pipeline
-----------------------------

A pipeline provides a non-persistent instance of the model for running
inference. When the script running this code exits, the model will also be
destroyed. The pipeline is ideal for doing quick tests or in cases where the
best performance is not necessary.

Copy the following code block into an ``example.py`` file on your local machine.
Run it with ``deepspeed --num_gpus <num of GPUs> example.py``.

.. code-block:: python

    import mii
    pipe = mii.pipeline("mistralai/Mistral-7B-v0.1")
    response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
    for r in response:
        print(r.generated_text)

.. note::

   Depending on your internet connection, the download of model weights could
   take a few minutes. If you wish to try a smaller model, replace
   ``mistralai/Mistral-7B-v0.1`` with ``facebook/opt-125m`` in the above code.

If the code successfully runs, you should see the generated text printed in your terminal.

Run a Persistent Deployment
---------------------------

In contrast the pipeline, deployments create a server process that persists
beyond the execution of the python script. These deployments are intended for
production use cases and allow for multiple clients to connect while providing
the best performance from DeepSpeed-FastGen.

Copy the following code block into a ``serve.py`` file on your local machine.
Run it with ``python serve.py``.

.. code-block:: python

    import mii
    mii.serve("mistralai/Mistral-7B-v0.1")

You should see logging messages indicating the server is starting and a final
log message of ``server has started on ports [50051]``.

Now copy the following code block into a ``client.py`` file on your local
machine. Run it with ``python client.py``.

.. code-block:: python

    import mii
    client = mii.client("mistralai/Mistral-7B-v0.1")
    response = client(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
    for r in response:
        print(r.generated_text)

If the code successfully runs, you should see the generated text printed in your
terminal. You can run this client script as many times (and from as many
different processes) as you like and the model deployment will remain active.

Finally copy the following code block into a ``terminate.py`` file on your local
machine. Run it with ``python terminate.py``.

.. code-block:: python

    import mii
    client = mii.client("mistralai/Mistral-7B-v0.1")
    client.terminate_server()

This will shutdown the model deployment and free GPU memory.
