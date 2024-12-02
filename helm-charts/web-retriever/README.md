# web-retriever

Helm chart for deploying Web Retriever microservice.

Web retriever depends on tei, you should set TEI_EMBEDDING_ENDPOINT endpoints before start.

## (Option1): Installing the chart separately

First, you need to install the tei chart, please refer to the [tei](../tei) chart for more information.

After you've deployted the tei chart successfully, please run `kubectl get svc` to get the tei service endpoint, i.e `http://tei`.

To install the web-retriever chart, run the following:

```console
cd GenAIInfra/helm-charts/common/web-retriever
helm dependency update
export TEI_EMBEDDING_ENDPOINT="http://tei"
export GOOGLE_API_KEY="yourownkey"
export GOOGLE_CSE_ID="yourownid"
helm install web-retriever . --set TEI_EMBEDDING_ENDPOINT=${TEI_EMBEDDING_ENDPOINT} --set GOOGLE_API_KEY=${GOOGLE_API_KEY} --set GOOGLE_CSE_ID=${GOOGLE_CSE_ID}
```

## (Option2): Installing the chart with dependencies automatically

```console
cd GenAIInfra/helm-charts/common/web-retriever
helm dependency update
export GOOGLE_API_KEY="yourownkey"
export GOOGLE_CSE_ID="yourownid"
helm install web-retriever . --set tei.enabled=true --set GOOGLE_API_KEY=${GOOGLE_API_KEY} --set GOOGLE_CSE_ID=${GOOGLE_CSE_ID}
```

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are running.

Then run the command `kubectl port-forward svc/web-retriever 7077:7077` to expose the web-retriever service for access.

Open another terminal and run the following command to verify the service if working:

```console
your_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
curl http://localhost:7077/v1/web_retrieval \
  -X POST \
  -d "{\"text\":\"What is OPEA?\",\"embedding\":${your_embedding}}" \
  -H 'Content-Type: application/json'
```

## Values

| Key                    | Type   | Default                       | Description |
| ---------------------- | ------ | ----------------------------- | ----------- |
| image.repository       | string | `"opea/web-retriever-chroma"` |             |
| service.port           | string | `"7077"`                      |             |
| TEI_EMBEDDING_ENDPOINT | string | `""`                          |             |
| GOOGLE_API_KEY         | string | `""`                          |             |
| GOOGLE_CSE_ID          | string | `""`                          |             |
