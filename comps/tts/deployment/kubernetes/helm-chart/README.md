# tts

Helm chart for deploying tts microservice.

tts depends on speecht5, you should set TTS_ENDPOINT endpoints before start.

## (Option1): Installing the chart separately

First, you need to install the speecht5 chart, please refer to the [speecht5](../speecht5) chart for more information.

After you've deployted the speecht5 chart successfully, please run `kubectl get svc` to get the speecht5 service endpoint, i.e. `http://speecht5:7055`.

To install the tts chart, run the following:

```console
cd GenAIInfra/helm-charts/common/tts
export TTS_ENDPOINT="http://speecht5:7055"
helm dependency update
helm install tts . --set TTS_ENDPOINT=${TTS_ENDPOINT}
```

## (Option2): Installing the chart with dependencies automatically

```console
cd GenAIInfra/helm-charts/common/tts
helm dependency update
helm install tts . --set speecht5.enabled=true
```

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are running.

Then run the command `kubectl port-forward svc/tts 9088:9088` to expose the tts service for access.

Open another terminal and run the following command to verify the service if working:

```console
curl http://localhost:9088/v1/audio/speech \
  -XPOST \
  -d '{"text": "Who are you?"}' \
  -H 'Content-Type: application/json'
```

## Values

| Key              | Type   | Default      | Description |
| ---------------- | ------ | ------------ | ----------- |
| image.repository | string | `"opea/tts"` |             |
| service.port     | string | `"9088"`     |             |
| TTS_ENDPOINT     | string | `""`         |             |
