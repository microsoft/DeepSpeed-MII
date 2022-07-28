import os
import mii
import torch
import deepspeed
from deepspeed.inference.engine import InferenceEngine
from deepspeed import OnDevice

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME, cached_path, hf_bucket_url
from transformers.utils.hub import EntryNotFoundError
from transformers.modeling_utils import get_checkpoint_shard_files
'''
TODO: The following class and functions are non-optimal (i.e., hacky) solutions
to getting the Bloom models working and will be refactored in a future PR
'''


class BloomPipeline(object):
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


def get_checkpoint_files(pretrained_model_name_or_path):
    cache_dir = None
    is_sharded = False
    revision = None
    local_files_only = False

    filename = WEIGHTS_NAME
    archive_file = hf_bucket_url(pretrained_model_name_or_path,
                                 filename=filename,
                                 revision=revision)

    try:
        resolved_archive_file = cached_path(
            archive_file,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return [resolved_archive_file]

    except (EntryNotFoundError, FileNotFoundError):
        if filename == WEIGHTS_NAME:
            # Maybe the checkpoint is sharded, we try to grab the index name in this case.
            archive_file = hf_bucket_url(
                pretrained_model_name_or_path,
                filename=WEIGHTS_INDEX_NAME,
                revision=revision,
            )
            resolved_archive_file = cached_path(
                archive_file,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            is_sharded = True

    if is_sharded:
        # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            cache_dir=cache_dir,
            revision=revision
        )

        return resolved_archive_file


def _bloom_ckpt_json():
    mii_cache = mii.utils.mii_cache_path()
    return os.path.join(mii_cache, "bloom-checkpoints.json")


def write_checkponts_json(model_name):
    import io
    import json
    checkpoints_json = _bloom_ckpt_json()
    with io.open(checkpoints_json, 'w', encoding='utf-8') as f:
        checkpoint_files = get_checkpoint_files(model_name)
        #checkpoint_files = ['/data/bloom-mp/bloom-mp_04.pt', '/data/bloom-mp/bloom-mp_02.pt', '/data/bloom-mp/bloom-mp_07.pt', '/data/bloom-mp/bloom-mp_05.pt', '/data/bloom-mp/bloom-mp_00.pt', '/data/bloom-mp/bloom-mp_01.pt', '/data/bloom-mp/bloom-mp_03.pt', '/data/bloom-mp/bloom-mp_06.pt']
        data = {
            "type": "BLOOM-176B",
            "checkpoints": checkpoint_files,
            "version": 1.0
        }  #, "parallelization": "tp"}
        json.dump(data, f)


# TODO: This function is a hack for the Bloom models and will be replaced with a LargeModel provider code path
def load_hf_llm(model_path, model_name, task_name, mii_config):
    deepspeed.init_distributed('nccl')
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    with OnDevice(dtype=torch.float16, enabled=True):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model = model.eval()
    if local_rank == 0:
        write_checkponts_json(model_name)
    torch.distributed.barrier()
    inference_pipeline = BloomPipeline(model=model, tokenizer=tokenizer)
    return inference_pipeline
