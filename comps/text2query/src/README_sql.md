# Text-to-SQL Microservice

## 🚀 Start Microservice with Python（Option 1）

### Install Requirements

```bash
pip install -r requirements.txt
```

### Start PostgresDB Service

We will use [Chinook](https://github.com/lerocha/chinook-database) sample database as a default to test the Text-to-SQL microservice. Chinook database is a sample database ideal for demos and testing ORM tools targeting single and multiple database servers.

```bash
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=testpwd
export POSTGRES_DB=chinook

cd comps/text2query

docker run --name postgres-db --ipc=host -e POSTGRES_USER=${POSTGRES_USER} -e POSTGRES_HOST_AUTH_METHOD=trust -e POSTGRES_DB=${POSTGRES_DB} -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} -p 5442:5432 -d -v ./chinook.sql:/docker-entrypoint-initdb.d/chinook.sql postgres:latest
```

### Start TGI Service

```bash
export HF_TOKEN=${HF_TOKEN}
export LLM_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
export TGI_PORT=8008

docker run -d --name="text2query-tgi-endpoint" --ipc=host -p $TGI_PORT:80 -v ./data:/data --shm-size 1g -e HF_TOKEN=${HF_TOKEN} -e model=${LLM_MODEL_ID} ghcr.io/huggingface/text-generation-inference:2.4.1 --model-id $LLM_MODEL_ID
```

### Verify the TGI Service

```bash
export your_ip=$(hostname -I | awk '{print $1}')
curl http://${your_ip}:${TGI_PORT}/generate \
  -X POST \
  -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}' \
  -H 'Content-Type: application/json'
```

### Setup Environment Variables

```bash
export TGI_LLM_ENDPOINT="http://${your_ip}:${TGI_PORT}"
export TEXT2QUERY_COMPONENT_NAME="OPEA_TEXT2QUERY_SQL"
```

### Start Text-to-SQL Microservice with Python Script

Start Text-to-SQL microservice with below command.

```bash
python3 opea_text2query_microservice.py
```

---

## 🚀 Start Microservice with Docker (Option 2)

### Start PostGreSQL Database Service

Please refer to section [Start PostgresDB Service](#start-postgresdb-service)

### Start TGI Service

Please refer to section [Start TGI Service](#start-tgi-service)

### Setup Environment Variables

```bash
export TGI_LLM_ENDPOINT="http://${your_ip}:${TGI_PORT}"
export TEXT2QUERY_COMPONENT_NAME="OPEA_TEXT2QUERY_SQL"
```

### Build Docker Image

```bash
cd GenAIComps/
docker build -t opea/text2query-sql:latest -f comps/text2query/src/Dockerfile .
```

### Run Docker with CLI (Option A)

```bash
export TGI_LLM_ENDPOINT="http://${your_ip}:${TGI_PORT}"

docker run  --runtime=runc --name="comps-langchain-text2query"  -p 9097:9097 --ipc=host -e llm_endpoint_url=${TGI_LLM_ENDPOINT} opea/text2query-sql:latest
```

### Run via docker compose (Option B)

#### Setup Environment Variables.

```bash
export TGI_LLM_ENDPOINT=http://${your_ip}:${TGI_PORT}
export HF_TOKEN=${HF_TOKEN}
export LLM_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=testpwd
export POSTGRES_DB=chinook
export LLM_ENDPOINT_PORT=${TGI_PORT}
export host_ip=${your_ip}
```

#### Start the services.

- Xeon CPU

```bash
cd comps/text2query/deployment/docker_compose
docker compose -f compose.yaml up text2query-sql -d
```

- Gaudi2 HPU

```bash
cd comps/text2sql/deployment/docker_compose
docker compose -f compose.yaml up text2query-sql-gaudi -d
```

---

## ✅ Invoke the microservice.

The Text-to-SQL microservice exposes the following API endpoints:

- Execute SQL Query from input text

  ```bash
  CONN_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${host_ip}:5442/${POSTGRES_DB}"
  curl http://${your_ip}:9097/v1/text2query\
          -X POST \
          -d '{"query": "Find the total number of Albums.", "conn_type": "sql", "conn_url": "'${CONN_URL}'", "conn_user": "'${POSTGRES_USER}'", "conn_password": "'${POSTGRES_PASSWORD}'", "conn_dialect": "postgresql" }' \
          -H 'Content-Type: application/json'
  ```
