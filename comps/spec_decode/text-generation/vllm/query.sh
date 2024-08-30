# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

your_ip="0.0.0.0"

curl http://${your_ip}:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "facebook/opt-6.7b",
  "prompt": "What is Deep Learning?",
  "max_tokens": 32,
  "temperature": 0
  }'

# ##query microservice
# curl http://${your_ip}:5000/v1/spec_decode/completions \
#   -X POST \
#   -d '{"query":"What is Deep Learning?","max_new_tokens":17,"top_p":0.95,"temperature":0.01,"streaming":false}' \
#   -H 'Content-Type: application/json'
