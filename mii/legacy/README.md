<!-- [![Build Status](https://github.com/microsoft/deepspeed-mii/workflows/Build/badge.svg)](https://github.com/microsoft/DeepSpeed-MII/actions) -->
[![Formatting](https://github.com/microsoft/DeepSpeed-MII/actions/workflows/formatting.yml/badge.svg)](https://github.com/microsoft/DeepSpeed-MII/actions/workflows/formatting.yml)
[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/Microsoft/DeepSpeed/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/deepspeed-mii.svg)](https://pypi.org/project/deepspeed-mii/)
<!-- [![Documentation Status](https://readthedocs.org/projects/deepspeed/badge/?version=latest)](https://deepspeed.readthedocs.io/en/latest/?badge=latest) -->

<div align="center">
 <img src="docs/images/mii-white.svg#gh-light-mode-only" width="400px">
 <img src="docs/images/mii-dark.svg#gh-dark-mode-only" width="400px">
</div>

## Latest News

* [2022/11] [Stable Diffusion Image Generation under 1 second w. DeepSpeed MII](examples/benchmark/txt2img)
* [2022/10] [Announcing DeepSpeed Model Implementations for Inference (MII)](https://www.deepspeed.ai/2022/10/10/mii.html)

# Contents

<!-- toc -->

- [DeepSpeed MII](#deepspeed-model-implementations-for-inference)
- [How does MII work?](#how-does-mii-work)
- [Supported Models and Tasks](#supported-models-and-tasks)
- [MII-Public and MII-Azure](#mii-public-and-mii-azure)
- [Getting started with MII](#getting-started-with-mii)
- [Quantifying Latency and Cost Reduction](#quantifying-latency-and-cost-reduction)
- [Community Tutorials](#community-tutorials)

<!-- tocstop -->

# DeepSpeed Model Implementations for Inference

![hero dark](docs/images/hero-dark.png#gh-dark-mode-only)
![hero light](docs/images/hero-transparent.png#gh-light-mode-only)

The Deep Learning (DL) open-source community has seen tremendous growth in the last few months. Incredibly powerful text generation models such as the Bloom 176B, or image generation model such as Stable Diffusion are now available to anyone with access to a handful or even a single GPU through platforms such as Hugging Face. While open sourcing has democratized access to AI capabilities, their application is still restricted by two critical factors: inference latency and cost.

There has been significant progress in system optimizations for DL model inference that can drastically reduce both latency and cost, but those are not easily accessible. A main reason for this limited accessibility is that the DL model inference landscape is diverse with models varying in size, architecture, system performance characteristics, hardware requirements, etc. Identifying the appropriate set of system optimizations applicable to a given model and applying them correctly is often beyond the scope of most data scientists, making low latency and low-cost inference mostly inaccessible.

DeepSpeed-MII is a new open-source python library from DeepSpeed, aimed towards making low-latency, low-cost inference of powerful models not only feasible but also easily accessible.

* MII offers access to highly optimized implementation of thousands of widely used DL models.
* MII supported models achieve significantly lower latency and cost compared to their original implementation. For example, MII reduces the latency of Big-Science Bloom 176B model by 5.7x, while reducing the cost by over 40x. Similarly, it reduces the latency and cost of deploying Stable Diffusion by 1.9x. See more details for [an exhaustive latency and cost analysis of MII](#quantifying-latency-and-cost-reduction).
* To enable low latency/cost inference, MII leverages an extensive set of optimizations from DeepSpeed-Inference such as deepfusion for transformers, automated tensor-slicing for multi-GPU inference, on-the-fly quantization with ZeroQuant, and several others (see our [blog post](https://www.deepspeed.ai/2022/10/10/mii.html) for more details).
* With state-of-the-art performance, MII supports low-cost deployment of these models both on-premises and on Azure via AML with just a few lines of codes.

# How does MII work?

![Text Generation Models](docs/images/mii-arch.png)

*Figure 1: MII Architecture, showing how MII automatically optimizes OSS models using DS-Inference before deploying them on-premises using GRPC, or on Microsoft Azure using AML Inference.*

Under-the-hood MII is powered by [DeepSpeed-Inference](https://arxiv.org/abs/2207.00032). Based on model type, model size, batch size, and available hardware resources, MII automatically applies the appropriate set of system optimizations from DeepSpeed-Inference to minimize latency and maximize throughput. It does so by using one of many pre-specified model injection policies, that allows MII and DeepSpeed-Inference to identify the underlying PyTorch model architecture and replace it with an optimized implementation (see *Figure A*). In doing so, MII makes the expansive set of optimizations in DeepSpeed-Inference automatically available for thousands of popular models that it supports.


# Supported Models and Tasks

MII currently supports over 50,000 models across a range of tasks such as text-generation, question-answering, text-classification. The models accelerated by MII are available through multiple open-sourced model repositories such as Hugging Face, FairSeq, EluetherAI, etc. We support dense models based on Bert, Roberta or GPT architectures ranging from few hundred million parameters to tens of billions of parameters in size. We continue to expand the list with support for massive hundred billion plus parameter dense and sparse models coming soon.

MII model support will continue to grow over time, check back for updates! Currently we support the following Hugging Face Transformers model families:

model family | size range | ~model count
------ | ------ | ------
[llama](https://huggingface.co/models?other=llama) | 7B - 65B | 1,500
[bloom](https://huggingface.co/models?other=bloom) | 0.3B - 176B | 480
[stable-diffusion](https://huggingface.co/models?other=stable-diffusion) | 1.1B | 3,700
[opt](https://huggingface.co/models?other=opt) | 0.1B - 66B | 460
[gpt\_neox](https://huggingface.co/models?other=gpt_neox) | 1.3B - 20B | 850
[gptj](https://huggingface.co/models?other=gptj) | 1.4B - 6B | 420
[gpt\_neo](https://huggingface.co/models?other=gpt_neo) | 0.1B - 2.7B | 700
[gpt2](https://huggingface.co/models?other=gpt2) | 0.3B - 1.5B | 11,900
[xlm-roberta](https://huggingface.co/models?other=xlm-roberta) | 0.1B - 0.3B | 4,100
[roberta](https://huggingface.co/models?other=roberta) | 0.1B - 0.3B | 8,700
[distilbert](https://huggingface.co/models?other=distilbert) | 0.1B - 0.3B | 4,700
[bert](https://huggingface.co/models?other=bert) | 0.1B - 0.3B | 23,600

<!--
SD param count:
text_encoder: 123060480
unet: 859520964
vae: 83653863
-->

<!--For a full set of models and tasks supported by MII, please see here (TODO: add reference to specific model classes we support)-->

# MII-Public and MII-Azure

MII can work with two variations of DeepSpeed-Inference. The first, referred to as ds-public, contains most of the DeepSpeed-Inference optimizations discussed here,  is also available via our open-source DeepSpeed library. The second referred to as ds-azure, offers tighter integration with Azure, and is available via MII to all Microsoft Azure customers. We refer to MII running the two DeepSpeed-Inference variants as MII-Public and MII-Azure, respectively.

While both variants offers significant latency and cost reduction over the open-sourced PyTorch baseline, the latter, offers additional performance advantage for generation based workloads. The full latency and cost advantage comparison with PyTorch baseline and across these two versions is available [here](#quantifying-latency-and-cost-reduction).

# Getting Started with MII

## Installation

We regularly push releases to [PyPI](https://pypi.org/project/deepspeed-mii/) and encourage users to install from there in most cases.

```bash
pip install deepspeed-mii
```

## Deploying MII-Public

MII-Public can be deployed on-premises or on any cloud offering with just a few lines of code. MII creates a lightweight GRPC server to support this form of deployment and provides a GRPC inference endpoint for queries.

Several deployment and query examples can be found here: [examples/local](examples/local)

As an example here is a deployment of the [bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m) model from Hugging Face:

**Deployment**
```python
import mii
mii_configs = {"tensor_parallel": 1, "dtype": "fp16"}
mii.deploy(task="text-generation",
           model="bigscience/bloom-560m",
           deployment_name="bloom560m_deployment",
           mii_config=mii_configs)
```

This will deploy the model onto a single GPU and start the GRPC server that can later be queried.

**Query**
```python
import mii
generator = mii.mii_query_handle("bloom560m_deployment")
result = generator.query({"query": ["DeepSpeed is", "Seattle is"]}, do_sample=True, max_new_tokens=30)
print(result)
```

The only required key is `"query"`, all other items outside the dictionary will be passed to `generate` as kwargs. For Hugging Face provided models you can find all possible arguments in their [documentation for generate](https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate).

**Shutdown Deployment**
```python
import mii
mii.terminate("bloom560m_deployment")
```

**Load balancing over multiple replicas**

You can launch a load balancer and multiple replica of MII servers.
When you specify a value for `replica_num`, `mii.deploy()` launches the load balancer server and `replica_num` number of replicas.
Note that each replica consists of `tensor_parallel` server processes that are deployed on the same server.

```python
mii_configs = {
...
    "tensor_parallel": tensor_parallel,
    "replica_num": replica_num,
    "hostfile": hostfile
}
mii.deploy(...
           mii_config=mii_configs,
           ...)
```

The client sends requests to the load balancer, which forwards them to the replicas, instead of sending requests to individual MII servers.
Currently, the load balancer implements a simple round-robin algorithm.
The load balancer acts as a simple proxy when `replica_num` is set to `1`.

`hostfile` is the path to hostfile used by DeepSpeed's launcher.
When hostfile is not specified, DeepSpeed-MII uses the default path `/job/hostfile`, which is defined for DeepSpeed.
See the [DeepSpeed's document](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) for the details.

**RESTful API support**

MII can enable users to call the inference service through RESTful APIs.
By setting `enable_restful_api` to `True`, `mii.deploy()` launches a gateway that accepts RESTful API.
The gateway can receive requests at `http://[HOST]:[PORT_FOR_RESTFUL_API]/mii/[DEPLOYMENT_NAME]`.

```python
mii_configs = {
...
    "enable_restful_api": True,
    "restful_api_port": PORT_FOR_RESTFUL_API,
...
}
mii.deploy(...
    deployment_name=DEPLOYMENT_NAME,
    mii_config=mii_configs)
```

**Non-persistent Deployment**

You can enable a non-persistent deployment which allows you to make queries without standing up a server. The non-persistent deployment acts as a simplified interface to DeepSpeed-inference for use cases that do not require creating a persistent model server process. Changing the `deployment_type` to `NON_PERSISTENT` in `mii.deploy(...)` will activate this option.

```python
...
mii.deploy(deployment_name = DEPLOYMENT_NAME,
	   deployment_type=mii.constants.DeploymentType.NON_PERSISTENT
	   ...
	   )

generator = mii.mii_query_handle(DEPLOYMENT_NAME)
result = generator.query({"query": ["DeepSpeed is", "Seattle is"]}, do_sample=True, max_new_tokens=30})

```

You can find a complete example [here]("https://github.com/microsoft/DeepSpeed-MII/tree/main/examples/non_persistent")

Any HTTP client can be used to call the APIs. An example of using curl is:
```bash
# Assume deployment_name and restful_api_port are set to bloom560m_deployment and 28080 respectively:
$ curl --header "Content-Type: application/json" --request POST  -d '{"request": {"query": ["Seattle is", "Bellevue is", "Redmond is"]}, "kwargs": {"do_sample": false, "max_new_tokens": 100}}' http://localhost:28080/mii/bloom560m_deployment
```

The code below is an example using Python.

```python
import requests
import json

# text_generation
url = 'http://localhost:28080/mii/bloom560m_deployment'
params = {"request": {"query": ["Seattle is", "Bellevue is", "Redmond is"]},
          "kwargs": {"do_sample": False, "max_new_tokens": 100}}

json_params = json.dumps(params)
response = requests.post(url, data=json_params, headers={
                         "Content-Type": "application/json"})
print(response.json())
```

## Deploying with MII-Azure

MII supports deployment on Azure via AML Inference. To enable this, MII generates AML deployment assets for a given model that can be deployed using the Azure-CLI, as shown in the code below. Furthermore, deploying on Azure, allows MII to leverage DeepSpeed-Azure as its optimization backend, which offers better latency and cost reduction than DeepSpeed-Public.

This deployment process is very similar to local deployments and we will modify the code from the local deployment example with the [bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m) model.

---
📌 **Note:**  MII-Azure has the benefit of supporting DeepSpeed-Azure for better latency and cost than DeepSpeed-Public for certain workloads. We are working to enable DeepSpeed-Azure automatically for all MII-Azure deployments in a near-term MII update. In the meantime, we are offering DeepSpeed-Azure as a preview release to MII-Azure users. If you have a MII-Azure deployment and would like to try DeepSpeed-Azure, please reach out to us at deepspeed-mii@microsoft.com to get access.

---

Several other AML deployment examples can be found here: [examples/aml](examples/aml)

**Setup**

To use MII on AML resources, you must have the Azure-CLI installed with an active login associated with your Azure resources. Follow the instructions below to get your local system ready for deploying on AML resources:

1. Install Azure-CLI. Follow the official [installation instructions](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli#install).
2. Run `az login` and follow the instructions to login to your Azure account. This account should be linked to the resources you plan to deploy on.
3. Set the default subscription with `az account set --subscription <YOUR-SUBSCRIPTION-ID>`. You can find your subscription ID in the "overview" tab on your resource group page from the Azure web portal.
4. Set the default resource group and workspace name with `az config defaults.group <YOUR-RESOURCE-GROUP> defaults.workspace <YOUR-WORKSPACE>`
5. Install the AML plugin for Azure-CLI with `az extension add --name ml`

**Deployment**
```python
import mii
mii_configs = {"tensor_parallel": 1, "dtype": "fp16"}
mii.deploy(task="text-generation",
           model="bigscience/bloom-560m",
           deployment_name="bloom560m-deployment",
           deployment_type=mii.constants.DeploymentType.AML,
           mii_config=mii_configs)
```

---
📌 **Note:** Running the `mii.deploy` with `deployment_type=mii.constants.DeploymentType.AML` will only generate the scripts to launch an AML deployment. You must also run the generated `deploy.sh` script to run on AML resources.

---

This will generate the scripts and configuration files necessary to deploy the model on AML using a single GPU. You can find the generated output at `./bloom560m-deployment_aml/`

When you are ready to run your deployment on AML resources, navigate to the newly created directory and run the deployment script:
```bash
cd ./bloom560m-deployment_aml/
bash deploy.sh
```

This script may take several minutes to run as it does the following:
- Downloads the model locally
- Creates a Docker Image with MII for your deployment
- Creates an AML online-endpoint for running queries
- Uploads and registers the model to AML
- Starts your deployment

---
📌 **Note:** Large models (e.g., `bigscience/bloom`) may cause a timeout when trying to upload and register the model to AML. In these cases, it is required to manually upload models to Azure blob storage with [AzCopy](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10). Instructions and automation of this step will be added soon.

---

**Query**
Once the deployment is running on AML, you can run queries by navigating to the online-endpoint that was created for this deployment (i.e., `bloom-560m-deployment-endpoint`) from the [AML web portal](https://ml.azure.com/endpoints). Select the "Test" tab at the top of the endpoint page and type your query into the text-box:
```
{"query": ["DeepSpeed is", "Seattle is"], "do_sample"=True, "max_new_tokens"=30}
```

The only required key is `"query"`, all other items in the dictionary will be passed to `generate` as kwargs. For Hugging Face provided models you can find all possible arguments in their [documentation for generate](https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate).

# Quantifying Latency and Cost Reduction

Inference workloads can be either latency critical, where the primary objective is to minimize latency, or cost sensitive, where the primary objective is to minimize cost. In this section, we quantify the benefits of using MII for both latency-critical and cost-sensitive scenarios.

## Latency Critical Scenarios

For latency-critical scenarios, where a small batch size of 1 is often used, MII can reduce the latency by up to 6x for a wide range of open-source models, across multiple tasks. More specifically, we show model latency reduction of [^overhead_details]:

1. Up to 5.7x for multi-GPU inference for text generation using massive models such as Big Science Bloom, Facebook OPT, and EluetherAI NeoX (*Figure 2 (left)*)

2. Up to 1.9x for image generation tasks model using Stable Diffusion (*Figure 2 (right)*)

3. Up to 3x for relatively smaller text generation models (up to 7B parameters) based on OPT, BLOOM, and GPT architectures, running on a single GPU (*Figures 3 and 4*)

4. Up to 9x for various text representation tasks like fill-mask, text classification, question answering, and token classification using RoBERTa- and BERT- based models (*Figures 5 and 6*).

[ ![multi gpu latency](docs/images/llm-latency-sd-latency.png) ](docs/images/llm-latency-sd-latency-zoom.png)
*Figure 2: (Left) Best achievable latency for large models. MII-Azure (int8) offers 5.7X lower latency compared to Baseline for Bloom-176B. (Right) Stable Diffusion text to image generation latency comparison.*

[ ![OPT and BLOOM Models](docs/images/opt-bloom.png) ](docs/images/opt-bloom.png)
*Figure 3: Latency comparison for OPT and BLOOM models. MII-Azure is up to 2.8x faster than baseline.*

[ ![GPT Models](docs/images/gpt.png) ](docs/images/mii/gpt.png)
*Figure 4: Latency comparison for GPT models. MII-Azure is up to 3x faster than baseline.*

[ ![Roberta Models](docs/images/roberta.png) ](docs/images/roberta.png)
*Figure 5: Latency comparison for RoBERTa models. MII offers up to 9x lower model latency and up to 3x lower end-to-end latency than baseline on several tasks and RoBERTa variants [^overhead_details].*

[ ![Bert Models](docs/images/bert.png) ](docs/images/bert.png)
*Figure 6: Latency comparison for BERT models. MII offers up to 8.9x lower model latency and up to 4.5x end-to-end latency across several tasks and BERT variants[^overhead_details].*

[^overhead_details]: The end-to-end latency of an inference workload is comprised of two components: i) actual model execution, and ii) pre-/post-processing before and after the model execution. MII optimizes the actual model execution but leaves the pre-/post-processing pipeline for future optimizations. We notice that text representation tasks have significant pre-/post-processing overhead (*Figures G and H*). We plan to address those in a future update.

## Cost Sensitive Scenarios

MII can significantly reduce the inference cost of very expensive language models like Bloom, OPT, etc. To get the lowest cost, we use a large batch size that maximizes throughput for both baseline and MII. Here we look at the cost reduction from MII using two different metrics: i) tokens generated per second per GPU, and ii) dollars per million tokens generated.

*Figures 7 and 8* show that MII-Public offers over 10x throughput improvement and cost reduction compared to the baseline, respectively. Furthermore, MII-Azure offers over 30x improvement in throughput and cost compared to the baseline.

[ ![tput large models](docs/images/tput-llms.png) ](docs/images/tput-llms.png)
*Figure 7: Throughput comparison per A100-80GB GPU for large models. MII-Public offers over 15x throughput improvement while MII-Azure offers over 40x throughput improvement.*

[ ![azure cost](docs/images/azure-cost.png) ](docs/images/azure-cost.png)
*Figure 8: Cost of generating 1 million tokens on Azure with different model types. MII-Azure reduces the cost of generation by over 40x.*

# Community Tutorials

* [DeepSpeed Deep Dive — Model Implementations for Inference (MII) (Heiko Hotz)](https://towardsdatascience.com/deepspeed-deep-dive-model-implementations-for-inference-mii-b02aa5d5e7f7)

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
