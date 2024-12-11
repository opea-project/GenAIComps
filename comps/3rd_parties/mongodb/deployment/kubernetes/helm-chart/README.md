# mongodb

Helm chart for deploying mongo DB service.

## Install the Chart

To install the chart, run the following:

```console
cd ${GenAIInfro_repo}/helm-charts/common
helm install mongodb mongodb
```

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all the mongo pods are runinng.

Then run the command `kubectl port-forward svc/mongodb 27017:27017` to expose the mongodb service for access.

Open another terminal and run the command `mongo --eval 'db.runCommand("ping").ok' localhost:27017/test --quiet ` to test mongodb access. The `mongo` command should return `1`.

## Values

| Key              | Type   | Default    | Description              |
| ---------------- | ------ | ---------- | ------------------------ |
| image.repository | string | `"mongo"`  |                          |
| image.tag        | string | `"7.0.11"` |                          |
| service.port     | string | `"27017"`  | The mongodb service port |
