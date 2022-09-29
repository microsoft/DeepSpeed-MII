import os
import torch
from deepspeed.inference.engine import InferenceEngine
from transformers import AutoTokenizer, AutoModelForCausalLM


class MIIPipeline(object):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, inputs, **kwargs):
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        if isinstance(self.model, InferenceEngine):
            self.model = self.model.module

        # expand proto list into py-list
        inputs = [i for i in inputs]
        tokens = self.tokenizer.batch_encode_plus(inputs,
                                                  return_tensors="pt",
                                                  padding=True)
        for t in tokens:
            if torch.is_tensor(tokens[t]):
                tokens[t] = tokens[t].to(f'cuda:{local_rank}')
        greedy_output = self.model.generate(**tokens, **kwargs)
        outputs = self.tokenizer.batch_decode(greedy_output, skip_special_tokens=True)

        # construct output to align w. HF pipeline
        output_dicts = []
        for output in outputs:
            output_dicts.append([{'generated_text': output}])

        return output_dicts


def hf_nopipe_provider(model_path, model_name, task_name, mii_config):
    local_rank = int(os.getenv('LOCAL_RANK', '0'))

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(f'cuda:{local_rank}')
    if mii_config.torch_dtype() == torch.half:
        model.half()

    mii_pipeline = MIIPipeline(model, tokenizer)
    return mii_pipeline
