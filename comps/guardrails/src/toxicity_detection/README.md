# Toxicity Detection Microservice

## Introduction

Toxicity Detection Microservice allows AI Application developers to safeguard user input and LLM output from harmful language in a RAG environment. By leveraging a smaller fine-tuned Transformer model for toxicity classification (e.g. DistilledBERT, RoBERTa, etc.), we maintain a lightweight guardrails microservice without significantly sacrificing performance.

This microservice uses [`Intel/toxic-prompt-roberta`](https://huggingface.co/Intel/toxic-prompt-roberta) that was fine-tuned on Gaudi2 with ToxicChat and Jigsaw Unintended Bias datasets.

Toxicity is defined as rude, disrespectful, or unreasonable language likely to make someone leave a conversation. This can include instances of aggression, bullying, targeted hate speech, or offensive language. For more information on labels see [Jigsaw Toxic Comment Classification Challenge](http://kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

## Environment Setup

### Clone OPEA GenAIComps and Setup Environment

Clone this repository at your desired location and set an environment variable for easy setup and usage throughout the instructions.

```bash
git clone https://github.com/opea-project/GenAIComps.git

export OPEA_GENAICOMPS_ROOT=$(pwd)/GenAIComps
```

Set the port that this service will use and the component name

```
export TOXICITY_DETECTION_PORT=9090
export TOXICITY_DETECTION_COMPONENT_NAME="OPEA_NATIVE_TOXICITY"
```

By default, this microservice uses `OPEA_NATIVE_TOXICITY` which invokes [`Intel/toxic-prompt-roberta`](https://huggingface.co/Intel/toxic-prompt-roberta), locally.

Alternatively, if you are using Prediction Guard, reset the following component name environment variable:

```
export TOXICITY_DETECTION_COMPONENT_NAME="PREDICTIONGUARD_TOXICITY_DETECTION"
```

### Set environment variables

## 🚀1. Start Microservice with Python（Option 1）

### 1.1 Install Requirements

```bash
cd $OPEA_GENAICOMPS_ROOT/comps/guardrails/src/toxicity_detection
pip install -r requirements.txt
```

### 1.2 Start Toxicity Detection Microservice with Python Script

```bash
python toxicity_detection.py
```

## 🚀2. Start Microservice with Docker (Option 2)

### 2.1 Build Docker Image

```bash
cd $OPEA_GENAICOMPS_ROOT
docker build \
    --build-arg https_proxy=$https_proxy \
    --build-arg http_proxy=$http_proxy \
    -t opea/guardrails-toxicity-detection:latest  \
    -f comps/guardrails/src/toxicity_detection/Dockerfile .
```

### 2.2.a Run Docker with Compose (Option A)

```bash
cd $OPEA_GENAICOMPS_ROOT/comps/guardrails/deployment/docker_compose
docker compose up -d guardrails-toxicity-detection-server
```

### 2.2.b Run Docker with CLI (Option B)

```bash
docker run -d --rm \
    --name="guardrails-toxicity-detection-server" \
    --runtime=runc  \
    -p ${TOXICITY_DETECTION_PORT}:9090 \
    --ipc=host \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e no_proxy=${no_proxy} \
     opea/guardrails-toxicity-detection:latest
```

## 🚀3. Get Status of Microservice

```bash
docker container logs -f guardrails-toxicity-detection-server
```

## 🚀4. Consume Microservice Pre-LLM/Post-LLM

Once microservice starts, users can use examples (bash or python) below to apply toxicity detection for both user's query (Pre-LLM) or LLM's response (Post-LLM)

**Bash:**

```bash
curl localhost:${TOXICITY_DETECTION_PORT}/v1/toxicity \
    -X POST \
    -d '{"text":"How to poison my neighbor'\''s dog without being caught?"}' \
    -H 'Content-Type: application/json'
```

Example Output:

```bash
"Violated policies: toxicity, please check your input."
```

**Python Script:**

```python
import requests
import json
import os

toxicity_detection_port = os.getenv("TOXICITY_DETECTION_PORT")
proxies = {"http": ""}
url = f"http://localhost:{toxicty_detection_port}/v1/toxicity"
data = {"text": "How to poison my neighbor'''s dog without being caught?"}


try:
    resp = requests.post(url=url, data=data, proxies=proxies)
    print(resp.text)
    resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
    print("Request successful!")
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
```
