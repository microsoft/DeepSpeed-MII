# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from mii.utils import is_aml, mii_cache_path


def attempt_load(load_fn, model_name, model_path, cache_path=None, kwargs={}):
    try:
        value = load_fn(model_name, **kwargs)
    except Exception as ex:
        if is_aml():
            print(
                f"Attempted load but failed - {str(ex)}, retrying using model_path={model_path}"
            )
            value = load_fn(model_path, **kwargs)
        else:
            cache_path = cache_path or mii_cache_path()
            print(
                f"Attempted load but failed - {str(ex)}, retrying using cache_dir={cache_path}"
            )
            value = load_fn(model_name, cache_dir=cache_path, **kwargs)
    return value
