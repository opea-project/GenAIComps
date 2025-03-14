# Text to graph triplet extractor

Creating graphs from text is about converting unstructured text into structured data is challenging.
It's gained significant traction with the advent of Large Language Models (LLMs), bringing it more into the mainstream. There are two main approaches to extract graph triplets depending on the types of LLM architectures like decode and encoder-decoder models.

## Decoder Models

Decoder-only models are faster during inference as they skip the encoding. This is ideal for tasks where the
input-output mapping is simpler or where multitasking is required. It is suitable for generating outputs based on
prompts or when computational efficiency is a priority. In certain cases, the decoder only models struggle with
tasks requiring deep contextual understanding or when input-output structures are highly heterogeneous.

## Encoder-decoder models

This microservice provides an encoder decoder architecture approach to graph triplet extraction. Models like REBEL, is based on the BART family/like model and fine-tuned for relation extraction and classification tasks. The approach works better when handling complex relations applications and data source. Encoder decoder models often achieve high performance on benchmarks due to their ability to encode contextual information effectively. It is suitable for tasks requiring detailed parsing of text into structured formats, such as knowledge graph construction from unstructured data.

# Features

Input text from a document or string(s) in text format and the graph triplets and nodes are identified.  
Subsequent processing needs to be done such as performing entity disambiguation to merge duplicate entities
before generating cypher code

## Implementation

The text-to-graph microservice able to extract from unstructured text in document, textfile, or string formats
The service is hosted in a docker. The text2graph extraction requires logic and LLMs to be hosted.
LLM hosting is done with TGI for Gaudi's and natively running on CPUs for CPU.

# 🚀1. Start Microservice with Docker

Option 1 running on CPUs

## Install Requirements

```bash
 pip install -r requirements.txt
```

## Environment variables : Configure LLM Parameters based on the model selected.

```
export LLM_ID=${LLM_ID:-"Babelscape/rebel-large"}
export SPAN_LENGTH=${SPAN_LENGTH:-"1024"}
export OVERLAP=${OVERLAP:-"100"}
export MAX_LENGTH=${MAX_NEW_TOKENS:-"256"}
export HUGGINGFACEHUB_API_TOKEN=""
export LLM_MODEL_ID=${LLM_ID}
export TGI_PORT=8008
```

##Echo env variables

```
echo "Extractor details"
echo LLM_ID=${LLM_ID}
echo SPAN_LENGTH=${SPAN_LENGTH}
echo OVERLAP=${OVERLAP}
echo MAX_LENGTH=${MAX_LENGTH}
```

### Start TGI Service

```bash
export HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
export LLM_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
export TGI_PORT=8008

docker run -d --name="text2graph-tgi-endpoint" --ipc=host -p $TGI_PORT:80 -v ./data:/data --shm-size 1g -e HF_TOKEN=${HUGGINGFACEHUB_API_TOKEN} -e model=${LLM_MODEL_ID} ghcr.io/huggingface/text-generation-inference:2.1.0 --model-id $LLM_MODEL_ID
```

### Verify the TGI Service

```bash
export your_ip=$(hostname -I | awk '{print $1}')
curl http://${your_ip}:${TGI_PORT}/generate \
  -X POST \
  -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}' \
  -H 'Content-Type: application/json'
```

### Setup Environment Variables to host TGI

```bash
export TGI_LLM_ENDPOINT="http://${your_ip}:${TGI_PORT}"
```

### Start Text2Graph Microservice with Docker

Command to build text2graph microservice

```bash
docker build -f Dockerfile -t user_name:graph_extractor ../../../
```

Command to launch text2graph microservice

```bash
docker run -i -t --net=host --ipc=host -p 8090 user_name:graph_extractor
```

The docker launches the text2graph microservice. To run it interactive.

# Validation and testing

## Text to triplets

Test directory is under GenAIComps/tests/text2graph/
There are two files in this directory.

- example_from_file.py : Example python script that downloads a text file and extracts triplets

- test_text2graph_opea.sh : The main script that checks for health and builds docker, extracts and generates triplets.

## Check if services are up

### Setup validation process

For set up use http://localhost:8090/docs for swagger documentation, list of commands, interactive GUI.
