# data-prep

Helm chart for deploying data-prep microservice.

data-prep will use redis and tei service, please specify the endpoints.

## (Option1): Installing the chart separately

First, you need to install the tei and redis-vector-db chart, please refer to the [tei](../tei/README.md) and [redis-vector-db](../redis-vector-db/README.md) for more information.

After you've deployted the tei and redis-vector-db chart successfully, please run `kubectl get svc` to get the service endpoint and URL respectively, i.e. `http://tei`, `redis://redis-vector-db:6379`.

To install data-prep chart, run the following:

```console
cd GenAIInfra/helm-charts/common/data-prep
export REDIS_URL="redis://redis-vector-db:6379"
export TEI_EMBEDDING_ENDPOINT="http://tei"
helm dependency update
helm install data-prep . --set REDIS_URL=${REDIS_URL} --set TEI_EMBEDDING_ENDPOINT=${TEI_EMBEDDING_ENDPOINT}
```

## (Option2): Installing the chart with dependencies automatically

```console
cd GenAIInfra/helm-charts/common/data-prep
helm dependency update
helm install data-prep . --set redis-vector-db.enabled=true --set tei.enabled=true

```

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are running.

Then run the command `kubectl port-forward svc/data-prep 6007:6007` to expose the data-prep service for access.

Open another terminal and run the following command to verify the service if working:

```console
curl http://localhost:6007/v1/dataprep  \
    -X POST \
    -H "Content-Type: multipart/form-data" \
    -F "files=@./README.md"
```

## Values

| Key                    | Type   | Default                 | Description |
| ---------------------- | ------ | ----------------------- | ----------- |
| image.repository       | string | `"opea/dataprep-redis"` |             |
| service.port           | string | `"6007"`                |             |
| REDIS_URL              | string | `""`                    |             |
| TEI_EMBEDDING_ENDPOINT | string | `""`                    |             |

## Milvus support

Refer to the milvus-values.yaml for milvus configurations.
