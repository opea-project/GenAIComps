# 🛢 Text-to-Cypher Microservice

The microservice enables a wide range of use cases, making it a versatile tool for businesses, researchers, and individuals alike. Users can generate queries based on natural language questions, enabling them to quickly retrieve relevant data from graph databases.

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

[Preparing Neo4J microservice](https://github.com/opea-project/GenAIComps/blob/main/comps/dataprep/src/README_neo4j_llamaindex.md)

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
docker build -t opea/text2cypher:latest -f comps/text2cypher/src/Dockerfile .
```

#### Run Docker with CLI (Option A)

```bash
docker run  --runtime=runc --name="comps-langchain-text2cypher"  -p 9097:8080 --ipc=host opea/text2cypher:latest
```

#### Run via docker compose (Option B)

##### Setup Environment Variables.

```bash
ip_address=$(hostname -I | awk '{print $1}')
export HF_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=neo4jtest
export NEO4J_URL="bolt://${ip_address}:7687"
```

##### Start the services.

- Xeon CPU

```bash
cd comps/text2cypher/deployment/docker_compose
docker compose -f compose.yaml up text2cypher -d
```

- Gaudi2 HPU

```bash
cd comps/text2cypher/deployment/docker_compose
docker compose -f compose.yaml up text2cypher-gaudi -d
```

---

### ✅ Invoke the microservice.

The Text-to-Cypher microservice exposes the following API endpoints:

- Execute Cypher Query from input text

  ```bash
  curl http://${your_ip}:9097/v1/text2cypher\
        -X POST \
        -d '{"input_text": "what are the symptoms for Diabetes?","conn_str": {"user": "'${NEO4J_USERNAME}'","password": "'${NEO4J_PASSWPORD}'","url": "'${NEO4J_URL}'" }}' \
        -H 'Content-Type: application/json'
  ```
