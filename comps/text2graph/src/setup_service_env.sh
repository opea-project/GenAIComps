# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#######################################################################
# Proxy
#######################################################################
export https_proxy=${https_proxy}
export http_proxy=${http_proxy}
export no_proxy=${no_proxy}
################################################################
# Configure LLM Parameters based on the model selected.
################################################################
export LLM_ID=${LLM_ID:-"Babelscape/rebel-large"}
export SPAN_LENGTH=${SPAN_LENGTH:-"1024"}
export OVERLAP=${OVERLAP:-"100"}
export MAX_LENGTH=${MAX_NEW_TOKENS:-"256"}
export HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
export HF_TOKEN=${HF_TOKEN}
export LLM_MODEL_ID=${LLM_ID}
export TGI_PORT=8008
export PYTHONPATH="/home/user/"
################################################################
### Echo env variables
################################################################
echo "Extractor details"
echo LLM_ID=${LLM_ID}
echo SPAN_LENGTH=${SPAN_LENGTH}
echo OVERLAP=${OVERLAP}
echo MAX_LENGTH=${MAX_LENGTH}
