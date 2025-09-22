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

Running on CPUs:

## Environment variables : Configure LLM Parameters based on the model selected.

```bash
export LLM_ID=${LLM_ID:-"Babelscape/rebel-large"}
export HF_TOKEN=""
export LLM_MODEL_ID=${LLM_ID}
source comps/text2query/src/integrations/graph/setup_service_env.sh
```

## Echo env variables

```bash
echo "Extractor details"
echo LLM_ID=${LLM_ID}
echo SPAN_LENGTH=${SPAN_LENGTH}
echo OVERLAP=${OVERLAP}
echo MAX_LENGTH=${MAX_LENGTH}
```

### Start Text2Graph Microservice with Docker

Command to build text2graph microservice

```bash
docker build --no-cache -t opea/text2query-graph:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/text2query/src/Dockerfile.graph .
```

Command to launch text2graph microservice

```bash
cd comps/text2query/deployment/docker_compose/compose.yaml
docker compose up text2query-graph -d
```

The docker launches the text2graph microservice. To run it interactive.

# Validation and testing

## Text to triplets

Test directory is under GenAIComps/tests/text2query/
There are two related files in this directory.

- example_from_file.py : Example python script that downloads a text file and extracts triplets

- test_text2query_graph.sh : The main script that checks for health and builds docker, extracts and generates triplets.
