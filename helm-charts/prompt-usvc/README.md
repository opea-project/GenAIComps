# prompt-usvc

Helm chart for deploying prompt-usvc microservice.

prompt-usvc will use mongo database service, please specify the endpoints.

## (Option1): Installing the chart separately

First, you need to install the mongodb chart, please refer to the [mongodb](../mongodb) for more information.

After you've deployted the mongodb chart successfully, run `kubectl get svc` to get the service endpoint and URL respectively, i.e. `mongodb:27017`.

To install prompt-usvc chart, run the following:

```console
cd GenAIInfra/helm-charts/common/prompt-usvc
export MONGO_HOST="mongodb"
export MONGO_PORT="27017"
helm dependency update
helm install prompt-usvc . --set MONGO_HOST=${MONGO_HOST} --set MONGO_PORT=${MONGO_PORT}
```

## (Option2): Installing the chart with dependencies automatically

```console
cd GenAIInfra/helm-charts/common/prompt-usvc
helm dependency update
helm install prompt-usvc . --set mongodb.enabled=true
```

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are running.

Then run the command `kubectl port-forward svc/prompt-usvc 6018:6018` to expose the prompt-usvc service for access.

Open another terminal and run the following command to verify the service if working:

```console
curl -X 'POST' \
  http://localhost:6018/v1/prompt/create \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"prompt_text": "test prompt", "user": "test"}';
```

## Values

| Key              | Type   | Default                       | Description |
| ---------------- | ------ | ----------------------------- | ----------- |
| image.repository | string | `"opea/promptregistry-mongo"` |             |
| service.port     | string | `"6018"`                      |             |
| MONGO_HOST       | string | `""`                          |             |
| MONGO_PORT       | string | `""`                          |             |
