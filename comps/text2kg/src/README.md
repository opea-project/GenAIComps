# Text to knowledge graph (text2kg) microservice

Text to Knowledge Graph (text2kg) Microservice enables the conversion of unstructured text into structured data by generating graph triplets. This process, which can be complex, has become more accessible with the rise of Large Language Models (LLMs), making it a mainstream solution for data extraction tasks. We are using a decoder-only model for this application's purpose.
This microservice can be run on cpu or hpu and instructions for the same are mentioned below.

## Decoder-Only Models

Decoder-only models are optimized for fast inference by skipping the encoding step. They work well for tasks where input-output mappings are relatively simple, or when multitasking is required. These models are ideal when computational efficiency and prompt-based output generation are priorities. However, decoder-only models may struggle with tasks that require deep contextual understanding or when input-output structures are highly complex or varied.

## Features

Input Formats: Accepts text from documents, text files, or strings\*.

Output: Answer to the query asked by the user.

## 🚀 1. Start individual microservices using docker cli (Option 1)

Update the environment_setup.sh file with your device and user information, and source it using -

```bash
source comps/text2kg/src/environment_setup.sh
```

If you skip this step, you can export variables related to individual services as mentioned in each of the microservices.

### 1. TGI

Refer to [this link](https://github.com/opea-project/GenAIComps/blob/main/comps/third_parties/tgi/README.md) to start and verify the TGI microservice.

### 2. Neo4J

Refer to [this link](https://github.com/opea-project/GenAIComps/blob/main/comps/third_parties/neo4j/src/README.md) to start and verify the neo4j microservice.

```bash
export DATA_DIRECTORY=$(pwd)
export ENTITIES="PERSON,PLACE,ORGANIZATION"
export RELATIONS="HAS,PART_OF,WORKED_ON,WORKED_WITH,WORKED_AT"
export VALIDATION_SCHEMA='{
    "PERSON": ["HAS", "PART_OF", "WORKED_ON", "WORKED_WITH", "WORKED_AT"],
    "PLACE": ["HAS", "PART_OF", "WORKED_AT"],
    "ORGANIZATION": ["HAS", "PART_OF", "WORKED_WITH"]
}'
```

### 3. Text2kg

```bash
cd comps/text2kg/src/
export TEXT2KG_PORT=8090
```

Build the text2kg docker image

```bash
docker build -f Dockerfile -t opea/text2kg:latest ../../../
```

Launch the docker container

```bash
docker run -i -t --net=host --ipc=host -p TEXT2KG_PORT -e HF_TOKEN=${HF_TOKEN} -e LLM_MODEL_ID=${LLM_MODEL_ID} opea/text2kg:latest -v data:/home/user/comps/text2kg/src/data /bin/bash
```

## 🚀 2. Start text2kg and dependent microservices with docker-compose (Option 2)

```bash
cd comps/text2kg/deployment/docker_compose/
```

Export service name and log path

```bash
export service_name="text2kg"
export LOG_PATH=$PWD
```

Export NEO4J variables - refer to section 1.2.b.
Launch using the following command to run on cpu

```bash
docker compose -f compose.yaml -f custom-override.yml up ${service_name}  -d > ${LOG_PATH}/start_services_with_compose.log
```

Launch using the following command to run on gaudi

```bash
docker compose -f compose.yaml up ${service_name}  -d > ${LOG_PATH}/start_services_with_compose.log
```

## 3. Check the service using API endpoint

```bash
curl -X 'POST' \
  'http://localhost:TEXT2KG_PORT/v1/text2kg?input_text=Who%20is%20paul%20graham%3F' \
  -H 'accept: application/json' \
  -d ''
```

- Make sure your input document/string has the necessary information that can be extracted.
