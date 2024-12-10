# tgi

Helm chart for deploying Hugging Face Text Generation Inference service.

## Installing the Chart

To install the chart, run the following:

```console
cd GenAIInfra/helm-charts/common
export MODELDIR=/mnt/opea-models
export MODELNAME="bigscience/bloom-560m"
export HFTOKEN="insert-your-huggingface-token-here"
helm install tgi tgi --set global.modelUseHostPath=${MODELDIR} --set LLM_MODEL_ID=${MODELNAME} --set global.HUGGINGFACEHUB_API_TOKEN=${HFTOKEN}
# To deploy on Gaudi enabled kubernetes cluster
# helm install tgi tgi --set global.modelUseHostPath=${MODELDIR} --set LLM_MODEL_ID=${MODELNAME} --set global.HUGGINGFACEHUB_API_TOKEN=${HFTOKEN} --values gaudi-values.yaml
```

By default, the tgi service will downloading the "bigscience/bloom-560m" which is about 1.1GB.

If you already cached the model locally, you can pass it to container like this example:

MODELDIR=/mnt/opea-models

MODELNAME="/data/models--bigscience--bloom-560m"

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are runinng.

Then run the command `kubectl port-forward svc/tgi 2080:80` to expose the tgi service for access.

Open another terminal and run the following command to verify the service if working:

```console
curl http://localhost:2080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}' \
    -H 'Content-Type: application/json'
```

## Values

| Key                             | Type   | Default                                           | Description                                                                                                                                                                                                           |
| ------------------------------- | ------ | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM_MODEL_ID                    | string | `"bigscience/bloom-560m"`                         | Models id from https://huggingface.co/, or predownloaded model directory                                                                                                                                              |
| global.HUGGINGFACEHUB_API_TOKEN | string | `insert-your-huggingface-token-here`              | Hugging Face API token                                                                                                                                                                                                |
| global.modelUseHostPath         | string | `"/mnt/opea-models"`                              | Cached models directory, tgi will not download if the model is cached here. The host path "modelUseHostPath" will be mounted to container as /data directory. Set this to null/empty will force it to download model. |
| image.repository                | string | `"ghcr.io/huggingface/text-generation-inference"` |                                                                                                                                                                                                                       |
| image.tag                       | string | `"1.4"`                                           |                                                                                                                                                                                                                       |
| autoscaling.enabled             | bool   | `false`                                           | Enable HPA autoscaling for the service deployment based on metrics it provides. See [HPA instructions](../../HPA.md) before enabling!                                                                                 |
| global.monitoring               | bool   | `false`                                           | Enable usage metrics for the service. Required for HPA. See [monitoring instructions](../../monitoring.md) before enabling!                                                                                           |
