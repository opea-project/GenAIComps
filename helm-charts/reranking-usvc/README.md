# reranking-usvc

Helm chart for deploying reranking microservice.

reranking-usvc depends on teirerank, set the TEI_RERANKING_ENDPOINT as teirerank endpoint.

## (Option1): Installing the chart separately

First, you need to install the teirerank chart, please refer to the [teirerank](../teirerank) chart for more information.

After you've deployted the teirerank chart successfully, please run `kubectl get svc` to get the tei service endpoint, i.e. `http://teirerank`.

To install the reranking-usvc chart, run the following:

```console
cd GenAIInfra/helm-charts/common/reranking-usvc
export TEI_RERANKING_ENDPOINT="http://teirerank"
helm dependency update
helm install reranking-usvc . --set TEI_RERANKING_ENDPOINT=${TEI_RERANKING_ENDPOINT}
```

## (Option2): Installing the chart with dependencies automatically

```console
cd GenAIInfra/helm-charts/common/reranking-usvc
helm dependency update
helm install reranking-usvc . --set teirerank.enabled=true
```

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are running.

Then run the command `kubectl port-forward svc/reranking-usvc 8000:8000` to expose the reranking-usvc service for access.

Open another terminal and run the following command to verify the service if working:

```console
curl http://localhost:8000/v1/reranking \
    -X POST \
    -d '{"initial_query":"What is Deep Learning?", "retrieved_docs": [{"text":"Deep Learning is not..."}, {"text":"Deep learning is..."}]}' \
    -H 'Content-Type: application/json'
```

## Values

| Key                    | Type   | Default                | Description |
| ---------------------- | ------ | ---------------------- | ----------- |
| image.repository       | string | `"opea/reranking-tgi"` |             |
| TEI_RERANKING_ENDPOINT | string | `""`                   |             |
| service.port           | string | `"8000"`               |             |
| global.monitoring      | bool   | `false`                |             |
