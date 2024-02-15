# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import io
from mii.legacy.utils import mii_cache_path


def supported_models_from_huggingface():
    return ["gpt2", "deepset/roberta-large-squad2"]


"""TODO make this more robust. If the pipeline has already been imported then
this might not work since the cache is set by the first import"""


def _download_hf_model_to_path(task, model_name, model_path):

    os.environ["TRANSFORMERS_CACHE"] = model_path
    from transformers import pipeline

    inference_pipeline = pipeline(task, model=model_name)


"""generic method that will allow downloading all models that we support.
Currently only supports HF models, but will be extended to support model checkpoints
from other sources"""


def download_model_and_get_path(task, model_name):

    model_path = os.path.join(mii_cache_path(), model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    if model_name in supported_models_from_huggingface():
        _download_hf_model_to_path(task, model_name, model_path)
    else:
        assert False, "Only models from HF supported so far"

    return model_path


class ImageResponse:
    def __init__(self, response):
        self._response = response
        self.nsfw_content_detected = response.nsfw_content_detected
        self._deserialized_images = None

    @property
    def images(self):
        if self._deserialized_images is None:
            from PIL import Image

            images = []
            for idx, img_bytes in enumerate(self._response.images):
                size = (self._response.size_w, self._response.size_h)
                img = Image.frombytes(self._response.mode, size, img_bytes)
                images.append(img)
            self._deserialized_images = images
        return self._deserialized_images


def convert_bytes_to_pil_image(image_bytes: bytes):
    """Converts bytes to a PIL Image object."""
    if not isinstance(image_bytes, bytes):
        return image_bytes

    from PIL import Image
    image = Image.open(io.BytesIO(image_bytes))
    return image
