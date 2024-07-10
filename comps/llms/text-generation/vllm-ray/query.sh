# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

your_ip="0.0.0.0"

##query vllm ray service
curl http://${your_ip}:8006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "facebook/opt-125m", "messages": [{"role": "user", "content": "How are you?"}]}'

##query microservice
curl http://${your_ip}:9000/v1/chat/completions \
  -X POST \
  -d '{"query":"What is Deep Learning?","max_new_tokens":17,"top_k":10,"top_p":0.95,"typical_p":0.95,"temperature":0.01,"repetition_penalty":1.03,"streaming":false}' \
  -H 'Content-Type: application/json'
