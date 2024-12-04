# speecht5

Helm chart for deploying speecht5 service.

## Installing the Chart

To install the chart, run the following:

```console
export MODELDIR=/mnt/opea-models
helm install speecht5 speecht5 --set global.modelUseHostPath=${MODELDIR}
```

## Verify

Use port-forward to access it from localhost.

```console
kubectl port-forward service/speecht5 1234:7055 &
curl http://localhost:1234/v1/tts \
  -XPOST \
  -d '{"text": "Who are you?"}' \
  -H 'Content-Type: application/json'
```

## Values

| Key              | Type   | Default           | Description |
| ---------------- | ------ | ----------------- | ----------- |
| image.repository | string | `"opea/speecht5"` |             |
| service.port     | string | `"7055"`          |             |
