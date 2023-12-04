Response Objects
================

Generated text from :doc:`pipeline` and :doc:`deployment` are wrapped in the
:class:`Response <mii.batching.data_classes.Response>` class.

.. autoclass::
    mii.batching.data_classes.Response
    :members:

Printing a :class:`Response <mii.batching.data_classes.Response>` object will
print only the ``generated_text`` attribute. Details about the generation can be
accessed as python attributes of the class:

.. code-block:: python

    responses = pipeline(["DeepSpeed is", "Seattle is"], max_length=128)
    for r in responses:
        print(f"generated length: {r.generated_length}, finish reason: {r.finish_reason}")

The reason that a text-generation request completed will be one of the values
found in the :class:`GenerationFinishReason
<mii.constants.GenerationFinishReason>` enum:

.. autoclass::
    mii.constants.GenerationFinishReason
    :inherited-members:
