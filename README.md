<!-- [![Build Status](https://github.com/microsoft/deepspeed-mii/workflows/Build/badge.svg)](https://github.com/microsoft/DeepSpeed-MII/actions) -->
[![Formatting](https://github.com/microsoft/DeepSpeed-MII/actions/workflows/formatting.yml/badge.svg)](https://github.com/microsoft/DeepSpeed-MII/actions/workflows/formatting.yml)
[![License MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Microsoft/DeepSpeed-MII/blob/master/LICENSE)
<!-- [![PyPI version](https://badge.fury.io/py/deepspeed.svg)](https://pypi.org/project/deepspeed/) -->
<!-- [![Documentation Status](https://readthedocs.org/projects/deepspeed/badge/?version=latest)](https://deepspeed.readthedocs.io/en/latest/?badge=latest) -->

<div align="center">
 <img src="docs/images/mii-white.svg#gh-light-mode-only" width="400px">
 <img src="docs/images/mii-dark.svg#gh-dark-mode-only" width="400px">
</div>

Model Implementations for Inference (MII) is library from DeepSpeed, designed to make low-latency, low-cost inference of powerful transformer models not only feasible but also easily accessible. It does so by offering access to highly optimized implementations of thousands of widely used DL models. In fact, straight out-of-box, MII supported models can be deployed on-premise with just a few lines of code.

**Note: MII is currently in a pre-release phase, this repo will be actively updated over the next several weeks with additional features, performance breakdowns, comparisons to other frameworks, etc.**

## How does MII work?

Under-the-hood MII is powered by DeepSpeed-Inference. Based on model type, model size, batch size, and available hardware resources, MII automatically applies the appropriate set of system optimizations from DeepSpeed-Inference to minimize latency and maximize thoughput. It does so using one of many pre-specified model injection policies, that allows DeepSpeed-Inference to identify the underlying PyTorch model architecture and replace it with an optimized implementation. This injection can replace a single GPU module with multi-GPU variations enabling models to run on single GPU device, or seamlessly scale to tens of GPUs for dense models and hundreds of GPUs for sparse models for lower latency and higher throughput.

MII makes the expansive set of optimizations in DeepSpeed-Inference easily accessible to its users by automatically integrating them to thousands of popular transformer models. For a full set of optimizations in DeepSpeed-Inference please see our paper: [DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](https://arxiv.org/abs/2207.00032).

## Supported Models and Tasks

MII currently supports over 20,000 models across a range of tasks such as text-generation, question-answering, text-classification. The models accelerated by MII are available through multiple open-sourced model repositories such as Hugging Face, FairSeq, EluetherAI, etc. We support dense models based on Bert, Roberta or GPT architectures ranging from few hundred million parameters to tens of billions of parameters in size. We continue to expand the list with support for massive hundred billion plus parameter dense and sparse models coming soon.

MII model support will continue to grow over time, check back for updates! Currently we support the following Hugging Face Transformers model families:

model family | size range | ~model count
------ | ------ | ------
[bloom](https://huggingface.co/models?other=bloom) | 0.3B - 176B | 120
[opt](https://huggingface.co/models?other=opt) | 0.1B - 66B | 70
[gpt\_neox](https://huggingface.co/models?other=gpt_neox) | 1.3B - 20B | 10
[gptj](https://huggingface.co/models?other=gptj) | 1.4B - 6B | 110
[gpt\_neo](https://huggingface.co/models?other=gpt_neo) | 0.1B - 2.7B | 260
[gpt2](https://huggingface.co/models?other=gpt2) | 0.3B - 1.5B | 7,100
[roberta](https://huggingface.co/models?other=roberta) | 0.1B - 0.3B | 4,200
[bert](https://huggingface.co/models?other=bert) | 0.1B - 0.3B | 12,500

<!--For a full set of models and tasks supported by MII, please see here (TODO: add reference to specific model classes we support)-->

## Getting Started with MII

### Installation

`pip install .` will install all dependencies required for deployment. A PyPI release of MII is coming soon.

### Deploying with MII-Public

MII allows supported models to be deployed with just a few lines of code on-premise.

Several deployment and query examples can be found here: [examples/local](https://github.com/microsoft/DeepSpeed-MII/tree/main/examples/local)

As an example here is a deployment of the [bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m) model from Hugging Face:

**Deployment**
```python
mii_configs = {"tensor_parallel": 1, "dtype": "fp16"}
mii.deploy(task="text-generation",
           model="bigscience/bloom-560m",
           deployment_name="bloom560m_deployment",
           mii_config=mii_configs)
```

This will deploy the model onto a single GPU and start the GRPC server that can later be queried.

**Query**
```python
generator = mii.mii_query_handle("bloom560m_deployment")
result = generator.query({"query": ["DeepSpeed is", "Seattle is"]}, do_sample=True, max_new_tokens=30)
print(result)
```

The only required key is `"query"`, all other items outside the dictionary will be passed to `generate` as kwargs. For Hugging Face provided models you can find all possible arguments in their [documentation for generate](https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate).

**Shutdown Deployment**
```python
mii.terminate("bloom560m_deployment")
```

### Deploying with MII-Azure

MII allows supported models to be deployed with just a few lines of code onto AzureML resources using AML Inference. This deployment process is very similar to local deployments and we will modify the code from the local deployment example with the [bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m) model.

---
📌 **Note:**  MII-Azure has the benefit of supporting DeepSpeed-Azure for better latency and cost than DeepSpeed-Public for certain workloads. We are working on enabling DeepSpeed-Azure automatically for MII-Azure deployment in near-term MII update. In the meantime, we offering DeepSpeed-Azure as a white glove service to our MII-Azure customers. If you have a MII-Azure deployment, please reach out to us at deepspeed-mii@microsoft.com to get access to DeepSpeed-Azure. 

---

Several other AML deployment examples can be found here: [examples/aml](https://github.com/microsoft/DeepSpeed-MII/tree/main/examples/aml)

**Setup**

To use MII on AML resources, you must have the Azure-CLI installed with an active login associated with your Azure resources. Follow the instructions below to get your local system ready for deploying on AML resources:

1. Install Azure-CLI. Follow the official [installation instructions](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli#install).
2. Run `az login` and follow the instructions to login to your Azure account. This account should be linked to the resources you plan to deploy on.
3. Set the default subscription with `az account set --subscription <YOUR-SUBSCRIPTION-ID>`. You can find your subscription ID in the "overview" tab on your resource group page from the Azure web portal.
4. Install the AML plugin for Azure-CLI with `az extension add --name ml`

**Deployment**
```python
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


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
