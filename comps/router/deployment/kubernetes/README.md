# Deploying the LLM Router Microservice on Kubernetes

This directory provides a **raw YAML-based deployment** for the LLM Router microservice using Kubernetes, following the OPEA `GenAIComps` structure.

## Overview

The Router microservice dynamically selects which model endpoint, should handle an incoming query based on routing logic. 

### Included Files

- **`router-configmap.yaml`**  
  Stores the top-level configuration (`config.yaml`) as a Kubernetes ConfigMap. This config defines:
  - The available models (via `model_map`)
  - Which routing controller to use (`controller_config_path`)

- **`router-service.yaml`**  
  Defines both the Deployment and the Service:
  - Mounts the config file from the ConfigMap
  - Injects runtime environment variables (e.g. `HF_TOKEN`, `OPENAI_API_KEY`)
  - Exposes the Router via a ClusterIP service on port `6000`

## Prerequisites

- A Kubernetes cluster (local or cloud-based)
- `kubectl` configured to point to your cluster
- Hugging Face and/or OpenAI API tokens, if your controller requires them

## Steps to Deploy

### 1. Set Up the ConfigMap

This holds the global configuration file that the router will use at runtime.

```bash
kubectl apply -f router-configmap.yaml
```

You can customize the file to switch controllers by changing:

```yaml
controller_config_path: "integrations/controllers/routellm/config.yaml"
```

You can also switch to:

```yaml
controller_config_path: "integrations/controllers/semantic_router/config.yaml"
```

### 2. Deploy the Router Microservice

```bash
kubectl apply -f router-service.yaml
```

This will:
- Launch a Deployment with a single pod running the Router container
- Expose it inside the cluster via a Service at port `6000`

> Note: Replace the placeholder API keys in `router-service.yaml` with real values or mount them via Kubernetes Secrets in production.

### 3. Test the Deployment

To access the router service locally:

```bash
kubectl port-forward svc/router-microservice-svc 6000:6000
```

Then call the `/v1/route` endpoint:

```bash
curl -X POST http://localhost:6000/v1/route \
     -H "Content-Type: application/json" \
     -d '{"text": "How do I reset my password?"}'
```

You should receive a response like:

```json
{
  "url": "http://some-inference-service:8000/weak"
}
```

### 4. Verify the Pod

Check that the router is up and running:

```bash
kubectl get pods
kubectl logs deployment/router-microservice
```

## Controller Config Access

Each controller (e.g., `routellm`, `semantic_router`) reads its own local `config.yaml`, which is **already included in the Docker image** at build time. The global `config.yaml` (from the ConfigMap) simply tells the microservice **which** controller to activate.


