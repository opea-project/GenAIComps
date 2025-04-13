#!/bin/bash
set -e

NAMESPACE="ogpt"

kubectl create configmap router-config \
  --from-file=config.yaml=config.yaml \
  --namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap routellm-config \
  --from-file=routellm_config.yaml=routellm_config.yaml \
  --namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap semanticrouter-config \
  --from-file=semantic_router_config.yaml=semantic_router_config.yaml \
  --namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# chmod +x create-configmaps.sh
# ./create-configmaps.sh
