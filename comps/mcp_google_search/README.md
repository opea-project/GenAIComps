# A MCP (Model-Context-Protocol) Google Search web retrieval component

This component provides a simple web retriever server and client based on the Model-Context-Protocol protocol. Please refer to [mcp](https://modelcontextprotocol.io/quickstart/server) for details.

## Build MCP Google Search web retriever image

```bash
cd ../../../..
docker build --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/mcp-google-search:latest -f comps/mcp_google_search/Dockerfile .
```


## Start the services
```bash
export host_ip=$(hostname -I | awk '{print $1}')

export OLLAMA_MODEL=qwen2.5-coder
export GOOGLE_API_KEY=$GOOGLE_API_KEY
export GOOGLE_CSE_ID=$GOOGLE_CSE_ID
export OLLAMA_ENDPOINT=http://${host_ip}:11434

systemctl stop ollama.service # docker will start ollama instead
docker compose up -d
```

## Run the client

```bash
docker exec -it mcp-google-search bash

cd src/google_search
python client.py mcp_google_search_server.py

# Output log will be like following
# USER_AGENT environment variable not set, consider setting it to identify your requests.

# Connected to server with tools: ['get-google-search-answer']

# MCP Client Started!
# Type your queries or 'quit' to exit.

# Query: search some latest sports news
# ...
# ...
```