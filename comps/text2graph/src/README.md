# Text to graph triplet extractor

Creating graphs from text is about converting unstructured text into structured data is challenging. 
It's gained significant traction with the advent of Large Language Models (LLMs), bringing it more into the mainstream.
There are approaches to extract graph triplets using different types of LLMs. 

##Encoder-decoder models 
such as REBEL, is based on the BART model and fine-tuned for relation extraction and classification tasks26.  
The other approach is Decoder only models. Depending on the applications and data source, the approach works better.
Encoder decoder models often achieve high performance on benchmarks due to their ability to encode contextual 
information effectively.  It is suitable for tasks requiring detailed parsing of text into structured formats, 
such as knowledge graph construction from unstructured data26.

##Decoder-Only Models
Decoder-only models are faster during inference as they skip the encoding. This is ideal for tasks where the 
input-output mapping is simpler or where multitasking is required.  It is suitable for generating outputs based on 
prompts or when computational efficiency is a priority.  In certain cases, the decoder only models struggle with 
tasks requiring deep contextual understanding or when input-output structures are highly heterogeneous.
This microservice provides an encoder decoder architecture approach to graph triplet extraction

---
# Features

**Provide text input and the graph triplets and nodes are identified**

## Implementation

The text-to-graph microservice able to extract from unstructured text 

#### 🚀 Start Microservice with Python（Option 1）

#### Install Requirements
```bash
pip install -r requirements.txt
```
---
### Environment variables : Configure LLM Parameters based on the model selected.
export LLM_ID=${LLM_ID:-"Babelscape/rebel-large"}
export SPAN_LENGTH=${SPAN_LENGTH:-"1024"}
export OVERLAP=${OVERLAP:-"100"}
export MAX_LENGTH=${MAX_NEW_TOKENS:-"256"}
export HUGGINGFACEHUB_API_TOKEN=""
export LLM_MODEL_ID=${LLM_ID}
export TGI_PORT=8008
---

---
###Echo env variables
echo "Extractor details"
echo LLM_ID=${LLM_ID}
echo SPAN_LENGTH=${SPAN_LENGTH}
echo OVERLAP=${OVERLAP}
echo MAX_LENGTH=${MAX_LENGTH}
---
#### Start TGI Service

```bash
export HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
export LLM_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
export TGI_PORT=8008

docker run -d --name="text2graph-tgi-endpoint" --ipc=host -p $TGI_PORT:80 -v ./data:/data --shm-size 1g -e HF_TOKEN=${HUGGINGFACEHUB_API_TOKEN} -e model=${LLM_MODEL_ID} ghcr.io/huggingface/text-generation-inference:2.1.0 --model-id $LLM_MODEL_ID
```

#### Verify the TGI Service

```bash
export your_ip=$(hostname -I | awk '{print $1}')
curl http://${your_ip}:${TGI_PORT}/generate \
  -X POST \
  -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}' \
  -H 'Content-Type: application/json'
```
#### Setup Environment Variables

```bash
export TGI_LLM_ENDPOINT="http://${your_ip}:${TGI_PORT}"
```

#### Start Text2Graph Microservice with Python Script

**Command to build text2graph microservice
docker build -f Dockerfile -t user_name:graph_extractor ../../../

**Command to launch text2graph microservice
docker run -i -t --net=host --ipc=host -p 8090 user_name:graph_extractor 

The docker launches the text2graph microservice.  To run it interactive 
```bash
python3 opea_text2graph_microservice.py
```
---

# Validation and testing

## Text to triplets
GenAIComps/tests/text2graph/

There are a few examples provided to help with the extraction. 
test_few_sentences.py generates triplets from couple of sentences. 
test_from_file.py download and feed a file. 
how to use it ? 
   python test_few_sentences.py
   python test_from_file.py

## Check if services are up
### Setup validation process 
   For set up use http://localhost:8090/docs for swagger documentation and list of commands 

