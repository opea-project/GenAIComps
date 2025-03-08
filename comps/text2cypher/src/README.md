# 🛢 Text-to-Cypher Microservice

The microservice enables a wide range of use cases, making it a versatile tool for businesses, researchers, and individuals alike. Users can generate queries based on natural language questions, enabling them to quickly retrieve relevant data from graph databases. This service executes locally on Intel Gaudi.

---

## 🛠️ Features

**Implement Cypher Query based on input text**: Transform user-provided natural language into Cypher queries, subsequently executing them to retrieve data from Graph databases.

---

## ⚙️ Implementation

The text-to-cypher microservice able to implement with various framework and support various types of Graph databases.

### 🔗 Utilizing Text-to-Cypher with Langchain framework

The follow guide provides set-up instructions and comprehensive details regarding the Text-to-Cypher microservices via LangChain. In this configuration, we will employ Neo4J DB as our example database to showcase this microservice.

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
export HF_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
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
