# chathistory-usvc

Helm chart for deploying chathistory-usvc microservice.

chathistory-usvc will use redis and tei service, please specify the endpoints.

## (Option1): Installing the chart separately

First, you need to install the mongodb chart, please refer to the [mongodb](../mongodb) for more information.

After you've deployted the mongodb chart successfully, run `kubectl get svc` to get the service endpoint and URL respectively, i.e. `mongodb:27017`.

To install chathistory-usvc chart, run the following:

```console
cd GenAIInfra/helm-charts/common/chathistory-usvc
export MONGO_HOST="mongodb"
export MONGO_PORT="27017"
helm dependency update
helm install chathistory-usvc . --set MONGO_HOST=${MONGO_HOST} --set MONGO_PORT=${MONGO_PORT}
```

## (Option2): Installing the chart with dependencies automatically

```console
cd GenAIInfra/helm-charts/common/chathistory-usvc
helm dependency update
helm install chathistory-usvc . --set mongodb.enabled=true
```

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are running.

Then run the command `kubectl port-forward svc/chathistory-usvc 6012:6012` to expose the chathistory-usvc service for access.

Open another terminal and run the following command to verify the service if working:

```console
curl -X 'POST' \
  http://localhost:6012/v1/chathistory/create \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"data": {"messages": "test Messages", "user": "test"}}'
```

## Values

| Key              | Type   | Default                           | Description |
| ---------------- | ------ | --------------------------------- | ----------- |
| image.repository | string | `"opea/chathistory-mongo-server"` |             |
| service.port     | string | `"6012"`                          |             |
| MONGO_HOST       | string | `""`                              |             |
| MONGO_PORT       | string | `""`                              |             |
