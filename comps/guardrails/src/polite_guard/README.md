# Politeness Guard Microservice

## Introduction

The Polite Guard Microservice allows AI application developers to ensure that user input and Large Language Model (LLM) outputs remain polite and respectful. By leveraging, [Polite Guard](https://huggingface.co/Intel/polite-guard), a fine-tuned Transformer model for politeness classification, this lightweight guardrails microservice ensures courteous interactions without significantly sacrificing performance, making it suitable for deployment on both Intel Gaudi and Xeon.

Politeness plays a crucial role in creating a positive and respectful environment. The service classifies text into four categories: _polite_, _somewhat polite_, _neutral_, and _impolite_. Any _impolite_ text is rejected, along with a score, ensuring that systems maintain a courteous tone.

More details about the Polite Guard model can be found [here](https://github.com/intel/polite-guard).

## 🚀1. Start Microservice with Python（Option 1）

### 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

### 1.2 Start Politeness Detection Microservice with Python Script

```bash
python opea_polite_guard_microservice.py
```

## 🚀2. Start Microservice with Docker (Option 2)

### 2.1 Prepare bias detection model

```bash
export HUGGINGFACEHUB_API_TOKEN=${YOUR_HF_TOKEN}
```

### 2.2 Build Docker Image

```bash
cd ../../../ # back to GenAIComps/ folder
docker build -t opea/guardrails-politeness-detection:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/guardrails/src/polite_guard/Dockerfile .
```

### 2.3 Run Docker Container with Microservice

```bash
docker run -d --rm --runtime=runc --name="guardrails-politeness-detection" -p 9092:9092 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN} -e HF_TOKEN=${HUGGINGFACEHUB_API_TOKEN} opea/guardrails-politeness-detection:latest
```

### 2.4 Get Status of Microservice

```bash
docker container logs -f guardrails-politeness-detection
```

### 2.5 Consume Microservice Pre-LLM/Post-LLM

Once microservice starts, users can use examples (bash or python) below to apply bias detection for both user's query (Pre-LLM) or LLM's response (Post-LLM)

**Bash:**

```bash
curl localhost:9092/v1/polite
    -X POST
    -d '{"text":"He is stupid"}'
    -H 'Content-Type: application/json'
```

Example Output:

```bash
"\nViolated policies: Impolite (score: 1.00), please check your input.\n"
```

**Python Script:**

```python
import requests
import json

proxies = {"http": ""}
url = "http://localhost:9092/v1/polite"
data = {"text": "He is stupid"}


try:
    resp = requests.post(url=url, data=data, proxies=proxies)
    print(resp.text)
    resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
    print("Request successful!")
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
```
