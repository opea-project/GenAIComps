# Deploy dataprep microservice on Kubernetes cluster

- You should have Helm (version >= 3.15) installed. Refer to the [Helm Installation Guide](https://helm.sh/docs/intro/install/) for more information.
- For more deployment options, refer to [helm charts README](https://github.com/opea-project/GenAIInfra/tree/main/helm-charts#readme).

## Deploy on Kubernetes with redis VectorDB

```
export HFTOKEN="insert-your-huggingface-token-here"
helm install data-prep oci://ghcr.io/opea-project/charts/data-prep --set global.HUGGINGFACEHUB_API_TOKEN=${HFTOKEN} -f redis-values.yaml
```

## Deploy on Kubernetes with milvus VectorDB

```
export HFTOKEN="insert-your-huggingface-token-here"
helm install data-prep oci://ghcr.io/opea-project/charts/data-prep --set global.HUGGINGFACEHUB_API_TOKEN=${HFTOKEN} -f milvus-values.yaml
```
