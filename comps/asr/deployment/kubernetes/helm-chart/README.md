# asr

Helm chart for deploying asr microservice.

asr depends on whisper, you should set ASR_ENDPOINT endpoints before start.

## (Option1): Installing the chart separately

First, you need to install the whisper chart, please refer to the [whisper](../whisper/README.md) chart for more information.

After you've deployted the whisper chart successfully, please run `kubectl get svc` to get the whisper service endpoint, i.e `http://whisper:7066`.

To install the asr chart, run the following:

```console
cd GenAIInfra/helm-charts/common/asr
export ASR_ENDPOINT="http://whisper:7066"
helm dependency update
helm install asr . --set ASR_ENDPOINT=${ASR_ENDPOINT}
```

## (Option2): Installing the chart with dependencies automatically

```console
cd GenAIInfra/helm-charts/common/asr
helm dependency update
helm install asr . --set whisper.enabled=true
```

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are running.

Then run the command `kubectl port-forward svc/asr 9099:9099` to expose the asr service for access.

Open another terminal and run the following command to verify the service if working:

```console
curl http://localhost:9099/v1/audio/transcriptions \
  -XPOST \
  -d '{"byte_str": "UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA"}' \
  -H 'Content-Type: application/json'
```

## Values

| Key              | Type   | Default      | Description |
| ---------------- | ------ | ------------ | ----------- |
| image.repository | string | `"opea/asr"` |             |
| service.port     | string | `"9099"`     |             |
| ASR_ENDPOINT     | string | `""`         |             |
