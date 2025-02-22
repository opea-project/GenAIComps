# Toxicity Detection Microservice

## Introduction

Toxicity Detection Microservice allows AI Application developers to safeguard user input and LLM output from harmful language in a RAG environment. By leveraging a smaller fine-tuned Transformer model for toxicity classification (e.g. DistillBERT, RoBERTa, etc.), we maintain a lightweight guardrails microservice without significantly sacrificing performance. This [article](https://huggingface.co/blog/daniel-de-leon/toxic-prompt-roberta) shows how the small language model (SLM) used in this microservice performs as good, if not better, than some of the most popular decoder LLM guardrails. This microservice uses [`Intel/toxic-prompt-roberta`](https://huggingface.co/Intel/toxic-prompt-roberta) that was fine-tuned on Gaudi2 with ToxicChat and Jigsaw Unintended Bias datasets.

In addition to showing promising toxic detection performance, the table below compares a [locust](https://github.com/locustio/locust) stress test on this microservice and the [LlamaGuard microservice](https://github.com/opea-project/GenAIComps/blob/main/comps/guardrails/src/guardrails/README.md#LlamaGuard). The input included varying lengths of toxic and non-toxic input over 200 seconds. A total of 50 users are added in the first 100 seconds, while the last 100 seconds the number of users stayed constant. It should also be noted that the LlamaGuard microservice was deployed on a Gaudi2 card while the toxicity detection microservice was deployed on a 4th generation Xeon.

| Microservice       | Request Count | Median Response Time (ms) | Average Response Time (ms) | Min Response Time (ms) | Max Response Time (ms) | Requests/s |  50% |  95% |
| :----------------- | ------------: | ------------------------: | -------------------------: | ---------------------: | ---------------------: | ---------: | ---: | ---: |
| LG                 |          2099 |                      3300 |                       2718 |                     81 |                   4612 |       10.5 | 3300 | 4600 |
| Toxicity Detection |          4547 |                       450 |                        796 |                     19 |                  10045 |       22.7 |  450 | 2500 |

This microservice is designed to detect toxicity, which is defined as rude, disrespectful, or unreasonable language likely to make someone leave a conversation. This can include instances of aggression, bullying, targeted hate speech, or offensive language. For more information on labels see [Jigsaw Toxic Comment Classification Challenge](http://kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

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
