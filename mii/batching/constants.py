# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# generate() kwargs
MAX_LENGTH_KWARG = "max_length"
MAX_NEW_TOKENS_KWARG = "max_new_tokens"
STREAM_KWARG = "stream"
IGNORE_EOS_KWARG = "ignore_eos"
TOP_P_KWARG = "top_p"
TOP_K_KWARG = "top_k"
TEMPERATURE_KWARG = "temperature"

# Default kwarg values
STREAM_DEFAULT = False
IGNORE_EOS_DEFAULT = False
TOP_P_DEFAULT = 0.9

# Processing method key names
TOP_K_NAME = "TopK"
TOP_P_NAME = "TopP"
TEMP_NAME = "Temp"
SAMPLER_NAME = "Sampler"
STOP_NAME = "Stop"
