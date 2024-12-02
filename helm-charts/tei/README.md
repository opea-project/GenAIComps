# tei

Helm chart for deploying Hugging Face Text Generation Inference service.

## Installing the Chart

To install the chart, run the following:

```console
cd ${GenAIInfro_repo}/helm-charts/common
export MODELDIR=/mnt/opea-models
export MODELNAME="BAAI/bge-base-en-v1.5"
helm install tei tei --set global.modelUseHostPath=${MODELDIR} --set EMBEDDING_MODEL_ID=${MODELNAME}
```

By default, the tei service will downloading the "BAAI/bge-base-en-v1.5" which is about 1.1GB.

If you already cached the model locally, you can pass it to container like this example:

MODELDIR=/mnt/opea-models

MODELNAME="/data/BAAI/bge-base-en-v1.5"

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are runinng.

Then run the command `kubectl port-forward svc/tei 2081:80` to expose the tei service for access.

Open another terminal and run the following command to verify the service if working:

```console
curl http://localhost:2081/embed -X POST -d '{"inputs":"What is Deep Learning?"}' -H 'Content-Type: application/json'
```

## Values

| Key                     | Type   | Default                                           | Description                                                                                                                                                                                                           |
| ----------------------- | ------ | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| EMBEDDING_MODEL_ID      | string | `"BAAI/bge-base-en-v1.5"`                         | Models id from https://huggingface.co/, or predownloaded model directory                                                                                                                                              |
| global.modelUseHostPath | string | `"/mnt/opea-models"`                              | Cached models directory, tei will not download if the model is cached here. The host path "modelUseHostPath" will be mounted to container as /data directory. Set this to null/empty will force it to download model. |
| image.repository        | string | `"ghcr.io/huggingface/text-embeddings-inference"` |                                                                                                                                                                                                                       |
| image.tag               | string | `"cpu-1.5"`                                       |                                                                                                                                                                                                                       |
| autoscaling.enabled     | bool   | `false`                                           | Enable HPA autoscaling for the service deployment based on metrics it provides. See [HPA instructions](../../HPA.md) before enabling!                                                                                 |
| global.monitoring       | bool   | `false`                                           | Enable usage metrics for the service. Required for HPA. See [monitoring instructions](../../monitoring.md) before enabling!                                                                                           |
