# teirerank

Helm chart for deploying Hugging Face Text Generation Inference service.

## Installing the Chart

To install the chart, run the following:

```console
cd ${GenAIInfro_repo}/helm-charts/common
export MODELDIR=/mnt/opea-models
export MODELNAME="BAAI/bge-reranker-base"
helm install teirerank teirerank --set global.modelUseHostPath=${MODELDIR} --set RERANK_MODEL_ID=${MODELNAME}
```

By default, the teirerank service will downloading the "BAAI/bge-reranker-base" which is about 1.1GB.

If you already cached the model locally, you can pass it to container like this example:

MODELDIR=/mnt/opea-models

MODELNAME="/data/BAAI/bge-reranker-base"

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are runinng.

Then run the command `kubectl port-forward svc/teirerank 2082:80` to expose the tei service for access.

Open another terminal and run the following command to verify the service if working:

```console
curl http://localhost:2082/rerank \
    -X POST \
    -d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}' \
    -H 'Content-Type: application/json'
```

## Values

| Key                     | Type   | Default                                           | Description                                                                                                                                                                                                                 |
| ----------------------- | ------ | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RERANK_MODEL_ID         | string | `"BAAI/bge-reranker-base"`                        | Models id from https://huggingface.co/, or predownloaded model directory                                                                                                                                                    |
| global.modelUseHostPath | string | `"/mnt/opea-models"`                              | Cached models directory, teirerank will not download if the model is cached here. The host path "modelUseHostPath" will be mounted to container as /data directory. Set this to null/empty will force it to download model. |
| image.repository        | string | `"ghcr.io/huggingface/text-embeddings-inference"` |                                                                                                                                                                                                                             |
| image.tag               | string | `"cpu-1.5"`                                       |                                                                                                                                                                                                                             |
| autoscaling.enabled     | bool   | `false`                                           | Enable HPA autoscaling for the service deployment based on metrics it provides. See [HPA instructions](../../HPA.md) before enabling!                                                                                       |
| global.monitoring       | bool   | `false`                                           | Enable usage metrics for the service. Required for HPA. See [monitoring instructions](../../monitoring.md) before enabling!                                                                                                 |
