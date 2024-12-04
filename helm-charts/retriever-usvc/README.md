# retriever-usvc

Helm chart for deploying Retriever microservice.

retriever-usvc depends on redis and tei, you should set these endpoints before start.

## (Option1): Installing the chart separately

First, you need to install the tei and redis-vector-db chart, refer to the [tei](../tei/README.md) and [redis-vector-db](../redis-vector-db/README.md) for more information.

After you've deployed the tei and redis-vector-db chart successfully, run `kubectl get svc` to get the service endpoint and URL respectively, i.e. `http://tei`, `redis://redis-vector-db:6379`.

To install retriever-usvc chart, run the following:

```console
cd GenAIInfra/helm-charts/common/retriever-usvc
export REDIS_URL="redis://redis-vector-db:6379"
export TEI_EMBEDDING_ENDPOINT="http://tei"
helm dependency update
helm install retriever-usvc . --set REDIS_URL=${REDIS_URL} --set TEI_EMBEDDING_ENDPOINT=${TEI_EMBEDDING_ENDPOINT}
```

## (Option2): Installing the chart with dependencies automatically

```console
cd GenAIInfra/helm-charts/common/retriever-usvc
helm dependency update
helm install retriever-usvc . --set tei.enabled=true --set redis-vector-db.enabled=true
```

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are running.

Then run the command `kubectl port-forward svc/retriever-usvc 7000:7000` to expose the retriever-usvc service for access.

Open another terminal and run the following command to verify the service if working:

```console
export your_embedding=$(python3 -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
curl http://localhost:7000/v1/retrieval  \
    -X POST \
    -d "{\"text\":\"What is the revenue of Nike in 2023?\",\"embedding\":${your_embedding}}" \
    -H 'Content-Type: application/json'
```

## Values

| Key                    | Type   | Default                | Description |
| ---------------------- | ------ | ---------------------- | ----------- |
| image.repository       | string | `"opea/retriever-tgi"` |             |
| service.port           | string | `"7000"`               |             |
| REDIS_URL              | string | `""`                   |             |
| TEI_EMBEDDING_ENDPOINT | string | `""`                   |             |
| global.monitoring      | bool   | `false`                |             |

## Milvus support

Refer to the milvus-values.yaml for milvus configurations.
