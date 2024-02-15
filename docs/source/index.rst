DeepSpeed-MII
=============

.. image:: ../images/mii-white.svg
   :width: 600

.. note::

   This project is under active development.


Introducing MII, an open-source Python library designed by DeepSpeed to
democratize powerful model inference with a focus on high-throughput, low
latency, and cost-effectiveness.

MII v0.1 introduced several features as part of our `DeepSpeed-FastGen release
<https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen>`_
such as blocked KV-caching, continuous batching, Dynamic SplitFuse, tensor
parallelism, and high-performance CUDA kernels to support fast high throughput
text-generation with LLMs. The latest version of MII delivers up to 2.5 times
higher effective throughput compared to leading systems such as vLLM. For
detailed performance results please see our `DeepSpeed-FastGen release blog
<https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen>`_
and the `latest DeepSpeed-FastGen blog
<https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen/2024-01-19>`_.

MII-Legacy
----------

We first `announced MII <https://www.deepspeed.ai/2022/10/10/mii.html>`_ in
2022. Since then, MII has undergone a large refactoring effort to bring support
of DeepSpeed-FastGen. MII-Legacy, which covers all prior releases up to v0.0.9,
provides support for running inference for a wide variety of language model
tasks. We also support accelerating `text2image models like Stable Diffusion
<https://github.com/Microsoft/DeepSpeed-MII/tree/main/mii/legacy/examples/benchmark/txt2img>`_.
For more details on our previous releases please see our `legacy APIs
<https://github.com/Microsoft/DeepSpeed-MII/tree/main/mii/legacy/>`_.


Contents
--------

.. toctree::
   :maxdepth: 1

   quick-start
   install
   api
   pipeline
   deployment
   response
   config
   rest
   parallelism
   replicas
