# 🛢 Text-to-Cypher Microservice

The microservice enables a wide range of use cases, making it a versatile tool for businesses, researchers, and individuals alike. Users can generate queries based on natural language questions, enabling them to quickly retrieve relevant data from graph databases. This service executes locally on Intel Gaudi.

---

## 🛠️ Features

**Implement Cypher Query based on input text**: Transform user-provided natural language into Cypher queries, subsequently executing them to retrieve data from Graph databases.

**MCP (Model Context Protocol) Support**: When enabled, this microservice can be discovered and used by AI agents through the MCP protocol, allowing seamless integration with AI-powered applications.

---

## ⚙️ Implementation

The text-to-cypher microservice able to implement with various framework and support various types of Graph databases.

### 🔗 Utilizing Text-to-Cypher with Langchain framework

The follow guide provides set-up instructions and comprehensive details regarding the Text-to-Cypher microservices via LangChain. In this configuration, we will employ Neo4J DB as our example database to showcase this microservice.

---

## 🤖 MCP (Model Context Protocol) Support

The text2cypher microservice now supports MCP, enabling AI agents to discover and use its natural language to Cypher query capabilities. This feature allows AI agents to interact with Neo4j databases through natural language queries.

### Enabling MCP Support

MCP support is controlled via the `ENABLE_MCP` environment variable:

```bash
export ENABLE_MCP=true  # Enable MCP support
```

When MCP is enabled, the service exposes an `/sse` endpoint for Server-Sent Events transport, allowing MCP clients to connect and discover available tools.

### Using with AI Agents

Once MCP is enabled, AI agents can:

1. Discover the text2cypher service through the OPEA MCP Tools Manager
2. Generate Cypher queries from natural language
3. Execute queries against Neo4j databases
4. Process and analyze graph data results

#### Example: Using with OpeaMCPToolsManager

```python
from comps.cores.mcp.manager import OpeaMCPToolsManager

# Initialize MCP manager
mcp_manager = OpeaMCPToolsManager()

# Add text2cypher service
await mcp_manager.add_sse_client("text2cypher", "http://localhost:9097/sse")

# List available tools
tools = await mcp_manager.list_tools()
# Will include: ['text2cypher']

# Use the text2cypher tool
result = await mcp_manager.call_tool(
    "text2cypher",
    {
        "input_text": "Find all movies directed by Christopher Nolan",
        "conn_str": {"url": "bolt://localhost:7687", "username": "neo4j", "password": "password"},
    },
)
```

### MCP Configuration in Docker Compose

When using Docker Compose, enable MCP by setting the environment variable:

```bash
export ENABLE_MCP=true
docker compose -f compose.yaml up text2cypher-gaudi -d
```

Or modify the compose file directly:

```yaml
environment:
  ENABLE_MCP: true
```

### Benefits of MCP Integration

- **AI Agent Discovery**: Agents can automatically discover and understand the text2cypher capabilities
- **Standardized Interface**: Uses the standard MCP protocol for consistent integration
- **Natural Language Queries**: Enables AI agents to query graph databases without knowing Cypher
- **Flexible Integration**: Works with any MCP-compatible AI agent framework

---

### Start Neo4J Service

### 🚀 Start Text2Cypher Microservice with Python（Option 1）

#### Install Requirements

```bash
pip install -r requirements.txt
```

#### Start Text-to-Cypher Microservice with Python Script

Start Text-to-Cypher microservice with below command.

```bash
python3 opea_text2cypher_microservice.py
```

---

### 🚀 Start Microservice with Docker (Option 2)

#### Build Docker Image

```bash
cd GenAIComps/
docker build -t opea/text2cypher:latest -f comps/text2cypher/src/Dockerfile.intel_hpu .
```

#### Run Docker with CLI (Option A)

```bash
docker run  --name="comps-langchain-text2cypher"  -p 9097:8080 --ipc=host opea/text2cypher:latest
```

#### Run via docker compose (Option B)

##### Setup Environment Variables.

```bash
ip_address=$(hostname -I | awk '{print $1}')
export HF_TOKEN=${HF_TOKEN}
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=neo4jtest
export NEO4J_URL="bolt://${ip_address}:7687"
export TEXT2CYPHER_PORT=11801
```

##### Start the services.

- Gaudi2 HPU

```bash
cd comps/text2cypher/deployment/docker_compose
docker compose -f compose.yaml up text2cypher-gaudi -d
```

---

### ✅ Invoke the microservice.

The Text-to-Cypher microservice exposes the following API endpoints:

- Execute Cypher Query with Pre-seeded Data and Schema:

  ```bash
  curl http://${ip_address}:${TEXT2CYPHER_PORT}/v1/text2cypher\
        -X POST \
        -d '{"input_text": "what are the symptoms for Diabetes?","conn_str": {"user": "'${NEO4J_USERNAME}'","password": "'${NEO4J_PASSWPORD}'","url": "'${NEO4J_URL}'" }}' \
        -H 'Content-Type: application/json'
  ```

- Execute Cypher Query with User Data and Schema:

Define customized cypher_insert statements:

```bash
export cypher_insert='
 LOAD CSV WITH HEADERS FROM "https://docs.google.com/spreadsheets/d/e/2PACX-1vQCEUxVlMZwwI2sn2T1aulBrRzJYVpsM9no8AEsYOOklCDTljoUIBHItGnqmAez62wwLpbvKMr7YoHI/pub?gid=0&single=true&output=csv" AS rows
 MERGE (d:disease {name:rows.Disease})
 MERGE (dt:diet {name:rows.Diet})
 MERGE (d)-[:HOME_REMEDY]->(dt)

 MERGE (m:medication {name:rows.Medication})
 MERGE (d)-[:TREATMENT]->(m)

 MERGE (s:symptoms {name:rows.Symptom})
 MERGE (d)-[:MANIFESTATION]->(s)

 MERGE (p:precaution {name:rows.Precaution})
 MERGE (d)-[:PREVENTION]->(p)
'
```

Pass the cypher_insert to the cypher2text service. The user can also specify whether to refresh the Neo4j database using the refresh_db option.

```bash
 curl http://${ip_address}:${TEXT2CYPHER_PORT}/v1/text2cypher \
        -X POST \
        -d '{"input_text": "what are the symptoms for Diabetes?", \
             "conn_str": {"user": "'${NEO4J_USERNAME}'","password": "'${NEO4J_PASSWPORD}'","url": "'${NEO4J_URL}'" } \
             "seeding": {"cypher_insert": "'${cypher_insert}'","refresh_db": "True" }}' \
        -H 'Content-Type: application/json'

```
