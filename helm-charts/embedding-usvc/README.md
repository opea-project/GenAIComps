# embedding-usvc

Helm chart for deploying embedding microservice.

embedding-usvc depends on TEI, set TEI_EMBEDDING_ENDPOINT.

## (Option1): Installing the chart separately

First, you need to install the tei chart, please refer to the [tei](../tei) chart for more information.

After you've deployted the tei chart successfully, please run `kubectl get svc` to get the tei service endpoint, i.e. `http://tei`.

To install the embedding-usvc chart, run the following:

```console
cd GenAIInfra/helm-charts/common/embedding-usvc
export TEI_EMBEDDING_ENDPOINT="http://tei"
helm dependency update
helm install embedding-usvc . --set TEI_EMBEDDING_ENDPOINT=${TEI_EMBEDDING_ENDPOINT}
```

## (Option2): Installing the chart with dependencies automatically

```console
cd GenAIInfra/helm-charts/common/embedding-usvc
helm dependency update
helm install embedding-usvc . --set tei.enabled=true
```

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are running.

Then run the command `kubectl port-forward svc/embedding-usvc 6000:6000` to expose the embedding-usvc service for access.

Open another terminal and run the following command to verify the service if working:

```console
curl http://localhost:6000/v1/embeddings \
    -X POST \
    -d '{"text":"hello"}' \
    -H 'Content-Type: application/json'
```

## Values

| Key                    | Type   | Default                | Description |
| ---------------------- | ------ | ---------------------- | ----------- |
| image.repository       | string | `"opea/embedding-tei"` |             |
| service.port           | string | `"6000"`               |             |
| TEI_EMBEDDING_ENDPOINT | string | `""`                   |             |
| global.monitoring      | bool   | `false`                |             |
