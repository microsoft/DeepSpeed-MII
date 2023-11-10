# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# generate() kwargs
MAX_LENGTH_KWARG = "max_length"
MAX_NEW_TOKENS_KWARG = "max_new_tokens"
MIN_NEW_TOKENS_KWARG = "min_new_tokens"
STREAM_KWARG = "stream"
IGNORE_EOS_KWARG = "ignore_eos"
TOP_P_KWARG = "top_p"
TOP_K_KWARG = "top_k"
TEMPERATURE_KWARG = "temperature"
RETURN_FULL_TEXT_KWARG = "return_full_text"
DO_SAMPLE_KWARG = "do_sample"
STOP_KWARG = "stop"

# Default kwarg values
MIN_NEW_TOKENS_DEFAULT = 0
STREAM_DEFAULT = False
IGNORE_EOS_DEFAULT = False
TOP_P_DEFAULT = 0.9
RETURN_FULL_TEXT_DEFAULT = False
DO_SAMPLE_DEFAULT = True

# Processing method key names
TOP_K_NAME = "TopK"
TOP_P_NAME = "TopP"
TEMP_NAME = "Temp"
SAMPLER_NAME = "Sampler"
STOP_NAME = "Stop"
