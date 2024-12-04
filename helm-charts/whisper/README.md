# whisper

Helm chart for deploying whisper service.

## Installing the Chart

To install the chart, run the following:

```console
export MODELDIR=/mnt/opea-models
export ASR_MODEL_PATH="openai/whisper-small"
helm install whisper whisper --set global.modelUseHostPath=${MODELDIR} --set ASR_MODEL_PATH=${ASR_MODEL_PATH}
```

## Verify

Use port-forward to access it from localhost.

```console
kubectl port-forward service/whisper 1234:7066 &
curl http://localhost:1234/v1/asr \
  -XPOST \
  -d '{"audio": "UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA"}' \
  -H 'Content-Type: application/json'
```

## Values

| Key              | Type   | Default          | Description |
| ---------------- | ------ | ---------------- | ----------- |
| image.repository | string | `"opea/whisper"` |             |
| service.port     | string | `"7066"`         |             |
