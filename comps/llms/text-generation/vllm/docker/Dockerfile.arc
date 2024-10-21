# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM intelanalytics/ipex-llm-serving-vllm-xpu-experiment:2.1.0b2

COPY comps/llms/text-generation/vllm/vllm_arc.sh /llm

RUN chmod +x /llm/vllm_arc.sh

ENTRYPOINT ["/llm/vllm_arc.sh"]
