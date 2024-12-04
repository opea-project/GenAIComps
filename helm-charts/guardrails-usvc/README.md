# guardrails-usvc

Helm chart for deploying LLM microservice.

guardrails-usvc depends on TGI, you should set TGI_LLM_ENDPOINT as tgi endpoint.

## (Option1): Installing the chart separately

First, you need to install the tgi chart, please refer to the [tgi](../tgi) chart for more information. Please use model `meta-llama/Meta-Llama-Guard-2-8B` during installation.

After you've deployted the tgi chart successfully, please run `kubectl get svc` to get the tgi service endpoint, i.e. `http://tgi`.

To install the chart, run the following:

```console
cd GenAIInfra/helm-charts/common/guardrails-usvc
export HFTOKEN="insert-your-huggingface-token-here"
export SAFETY_GUARD_ENDPOINT="http://tgi"
export SAFETY_GUARD_MODEL_ID="meta-llama/Meta-Llama-Guard-2-8B"
helm dependency update
helm install guardrails-usvc . --set global.HUGGINGFACEHUB_API_TOKEN=${HFTOKEN} --set SAFETY_GUARD_ENDPOINT=${SAFETY_GUARD_ENDPOINT} --set SAFETY_GUARD_MODEL_ID=${SAFETY_GUARD_MODEL_ID} --wait
```

## (Option2): Installing the chart with dependencies automatically

```console
cd GenAIInfra/helm-charts/common/guardrails-usvc
export HFTOKEN="insert-your-huggingface-token-here"
helm dependency update
helm install guardrails-usvc . --set global.HUGGINGFACEHUB_API_TOKEN=${HFTOKEN} --set tgi-guardrails.enabled=true --wait
```

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are running.

Then run the command `kubectl port-forward svc/guardrails-usvc 9090:9090` to expose the llm-uservice service for access.

Open another terminal and run the following command to verify the service if working:

```console
curl http://localhost:9090/v1/guardrails \
    -X POST \
    -d '{"text":"How do you buy a tiger in the US?","parameters":{"max_new_tokens":32}}' \
    -H 'Content-Type: application/json'
```

## Values

| Key                             | Type   | Default                              | Description                                                                                                                                                  |
| ------------------------------- | ------ | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| global.HUGGINGFACEHUB_API_TOKEN | string | `""`                                 | Your own Hugging Face API token                                                                                                                              |
| global.modelUseHostPath         | string | `"/mnt/opea-models"`                 | Cached models directory, tgi will not download if the model is cached here. The host path "modelUseHostPath" will be mounted to container as /data directory |
| image.repository                | string | `"opea/guardrails-usvc"`             |                                                                                                                                                              |
| service.port                    | string | `"9090"`                             |                                                                                                                                                              |
| SAFETY_GUARD_ENDPOINT           | string | `""`                                 | LLM endpoint                                                                                                                                                 |
| SAFETY_GUARD_MODEL_ID           | string | `"meta-llama/Meta-Llama-Guard-2-8B"` | Model ID for the underlying LLM service is using                                                                                                             |
