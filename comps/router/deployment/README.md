# Router Deployment

This folder contains **deployment assets** for running the Router microservice in various environments, including Docker Compose and Kubernetes.

## Contents

```
comps/router/deployment
├── README.md             # (this file)
├── docker_compose/
│   ├── compose.yaml
│   ├── config.yaml
│   ├── routellm_config.yaml
│   ├── semantic_router_config.yaml
│   └── deploy_router.sh
└── kubernetes/
    ├── config.yaml
    ├── create-configmaps.sh
    ├── router-deployment.yaml
    ├── router-service.yaml
    ├── routellm_config.yaml
    └── semantic_router_config.yaml
```

### 1. Docker Compose

- **`docker_compose/compose.yaml`**: Defines the Router container, volumes, and environment variables.  
- **`deploy_router.sh`**: Simple script that:
  1. Reads environment variables (Hugging Face token, OpenAI API key, etc.).
  2. Launches `compose.yaml`.
- **Configuration Files**: `config.yaml`, `routellm_config.yaml`, `semantic_router_config.yaml`.  
  - `config.yaml` typically references one of the specialized config files.

#### Usage Example

1. **Set up tokens**:
   ```
   export HF_TOKEN="your-hf-token"
   export OPENAI_API_KEY="your-openai-key"
   ```
2. **Run**:
   ```
   cd comps/router/deployment/docker_compose
   ./deploy_router.sh
   ```
   The router will be on `localhost:6000` by default.

### 2. Kubernetes

- **`router-deployment.yaml`**: A Deployment manifest that references:
  - `router-config` config map
  - `routellm-config` or `semanticrouter-config` config map
- **`router-service.yaml`**: Exposes the router as a Service (port 6000 by default).
- **`create-configmaps.sh`**: Creates three config maps in a namespace (default `ogpt`).

#### Usage Example

```
cd comps/router/deployment/kubernetes
./create-configmaps.sh
kubectl apply -f router-deployment.yaml
kubectl apply -f router-service.yaml

kubectl get pods
kubectl get svc
```

**Check logs**:
```
kubectl logs deploy/router -f
```

---

## Cleanup

**Docker Compose**
```
docker compose -f docker_compose/compose.yaml down
```

**Kubernetes**
```
kubectl delete -f router-deployment.yaml
kubectl delete -f router-service.yaml
kubectl delete configmap router-config routellm-config semanticrouter-config
```

---

## Next Steps

- For more info on the Router's code structure, see [../src/README.md](../src/README.md).
- To run automated tests, see [../tests/README.md](../tests/README.md).