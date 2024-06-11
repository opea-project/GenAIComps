# PII Detection Microservice

# 🚀1. Start Microservice with Python（Option 1）

## 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

## 1.2 Start LLM endpoint

TBD: Please refer to this [readme](../../../vectorstores/langchain/redis/README.md).

## 1.3 Setup Environment Variables

<!-- ```bash
export REDIS_URL="redis://${your_ip}:6379"
export INDEX_NAME=${your_index_name}
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=${your_langchain_api_key}
export LANGCHAIN_PROJECT="opea/gen-ai-comps:dataprep"
``` -->

## 1.4 Start PII Detection Microservice with Python Script

Start pii detection microservice with below command.

```bash
python pii_detection.py
```

# 🚀2. Start Microservice with Docker (Option 2)

## 2.1 Prepare PII detection model

## 2.1.1 use LLM endpoint

TBD

## 2.1.2 use NER model (default mode)

``` bash
mkdir -p pii/bigcode
apt install git-lfs
cd pii/bigcode; git clone https://{hf_username}:{hf_token}@huggingface.co/bigcode/starpii/; cd ../..
```

## 2.2 Setup Environment Variables

TBD

## 2.3 Build Docker Image

```bash
cd ../../../ # back to GenAIComps/ folder
docker build -t opea/guardrails-pii-detection:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/guardrails/pii_detection/docker/Dockerfile .
```

## 2.4 Run Docker with CLI

```bash
docker run -d --rm --runtime=runc --name="guardrails-pii-detection-endpoint" -p 6357:6357 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/guardrails-pii-detection:latest
```

> debug mode
```bash
docker run --rm --runtime=runc --name="guardrails-pii-detection-endpoint" -p 6357:6357 -v ./comps/guardrails/pii_detection/:/home/user/comps/guardrails/pii_detection/ --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/guardrails-pii-detection:latest
```

# 🚀3. Status Microservice

``` bash
docker container logs -f guardrails-pii-detection-endpoint
```

# 🚀4. Consume Microservice

Once microservice starts, user can use below script to invoke the microservice for pii detection.

``` python
import requests
import json

proxies = {"http": ""}
url = "http://localhost:6357/v1/dataprep"
urls = [
    "https://towardsdatascience.com/no-gpu-no-party-fine-tune-bert-for-sentiment-analysis-with-vertex-ai-custom-jobs-d8fc410e908b?source=rss----7f60cf5620c9---4"
]
payload = {"link_list": json.dumps(urls)}

try:
    resp = requests.post(url=url, data=payload, proxies=proxies)
    print(resp.text)
    resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
    print("Request successful!")
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
```
