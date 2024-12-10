# lvm-uservice

Helm chart for deploying LVM microservice.

lvm-uservice depends on TGI, you should set LVM_ENDPOINT as tgi endpoint.

## (Option1): Installing the chart separately

First, you need to install the tgi chart, please refer to the [tgi](../tgi) chart for more information.

After you've deployted the tgi chart successfully, please run `kubectl get svc` to get the tgi service endpoint, i.e. `http://tgi`.

To install the chart, run the following:

```console
cd GenAIInfra/helm-charts/common/lvm-uservice
export HFTOKEN="insert-your-huggingface-token-here"
export LVM_ENDPOINT="http://tgi"
helm dependency update
helm install lvm-uservice . --set global.HUGGINGFACEHUB_API_TOKEN=${HFTOKEN} --set LVM_ENDPOINT=${LVM_ENDPOINT} --wait
```

## (Option2): Installing the chart with dependencies automatically

```console
cd GenAIInfra/helm-charts/common/lvm-uservice
export HFTOKEN="insert-your-huggingface-token-here"
helm dependency update
helm install lvm-uservice . --set global.HUGGINGFACEHUB_API_TOKEN=${HFTOKEN} --set tgi.enabled=true --wait
```

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are running.

Then run the command `kubectl port-forward svc/lvm-uservice 9000:9000` to expose the lvm-uservice service for access.

Open another terminal and run the following command to verify the service if working:

```console
curl http://localhost:9000/v1/chat/completions \
    -X POST \
    -d '{"query":"What is Deep Learning?","max_tokens":17,"top_k":10,"top_p":0.95,"typical_p":0.95,"temperature":0.01,"repetition_penalty":1.03,"streaming":true}' \
    -H 'Content-Type: application/json'
```

## Values

| Key                             | Type   | Default          | Description                     |
| ------------------------------- | ------ | ---------------- | ------------------------------- |
| global.HUGGINGFACEHUB_API_TOKEN | string | `""`             | Your own Hugging Face API token |
| image.repository                | string | `"opea/lvm-tgi"` |                                 |
| service.port                    | string | `"9000"`         |                                 |
| LVM_ENDPOINT                    | string | `""`             | LVM endpoint                    |
| global.monitoring               | bool   | `false`          | Service usage metrics           |
