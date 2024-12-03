# OPEA <CHARTNAME> microservice

Helm chart for deploying OPEA example service.

## Installing the Chart

To install the chart, run the following:

```console

```

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are runinng and in ready state.

Then run the command `kubectl port-forward svc/example 8080:8080` to expose the tgi service for access.

Open another terminal and run the following command to verify the service if working:

```console

```

## Values

| Key                             | Type   | Default                              | Description            |
| ------------------------------- | ------ | ------------------------------------ | ---------------------- |
| global.HUGGINGFACEHUB_API_TOKEN | string | `insert-your-huggingface-token-here` | Hugging Face API token |
