# redis-vector-db

Helm chart for deploying Redis Vector DB service.

## Install the Chart

To install the chart, run the following:

```console
cd ${GenAIInfro_repo}/helm-charts/common
helm install redis-vector-db redis-vector-db
```

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all the redis pods are runinng.

Then run the command `kubectl port-forward svc/redis-vector-db 6379:6379` to expose the redis vector db service for access.

Open another terminal and run the command `redis-cli -h 127.0.0.1 -p 6379 ping` to access the redis vector db. The `redis-cli` command should return `PONG`.

## Values

| Key                          | Type   | Default               | Description            |
| ---------------------------- | ------ | --------------------- | ---------------------- |
| image.repository             | string | `"redis/redis-stack"` |                        |
| image.tag                    | string | `"7.2.0-v9"`          |                        |
| service.port (redis-service) | string | `"6379"`              | The redis-service port |
| service.port (redis-insight) | string | `"8001"`              | The redis-insight port |
