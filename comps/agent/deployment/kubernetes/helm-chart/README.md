# agent

Helm chart for deploying Agent microservice.

agent depends on LLM service, you should set llm_endpoint_url as LLM endpoint.

## Deploy

### Use external LLM endpoint

helm install agent oci://ghcr.io/opea-project/charts/agent --set llm_endpoint_url=${YOUR_LLM_ENDPOINT}

### Deploy with tgi

helm install agent oci://ghcr.io/opea-project/charts/agent --set tgi.enabled=True

### Deploy with vllm

helm install agent oci://ghcr.io/opea-project/charts/agent --set vllm.enabled=True

## Verify

To verify the installation, run the command `kubectl get pod` to make sure all pods are running.

Then run the command `kubectl port-forward svc/agent 9090:9090` to expose the agent service for access.

Open another terminal and run the following command to verify the service if working:

```console
curl http://localhost:9090/v1/chat/completions \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{"query":"What is OPEA?"}'
```

## Options

For global options, see Global Options.

| Key                             | Type   | Default                  | Description                     |
| ------------------------------- | ------ | ------------------------ | ------------------------------- |
| global.HUGGINGFACEHUB_API_TOKEN | string | `""`                     | Your own Hugging Face API token |
| image.repository                | string | `"opea/agent-langchain"` |                                 |
| service.port                    | string | `"9090"`                 |                                 |
| llm_endpoint_url                | string | `""`                     | LLM endpoint                    |
| global.monitoring               | bop;   | false                    | Service usage metrics           |
