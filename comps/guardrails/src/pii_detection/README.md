# PII Detection Microservice

## Introduction

In today's digital landscape, safeguarding personal information has become paramount, necessitating robust mechanisms to detect and protect personally identifiable information (PII). PII detection guardrails serve as essential tools in this endeavor, providing automated systems and protocols designed to identify, manage, and secure sensitive data. These guardrails leverage classical machine learning, LLMs, natural language processing (NLP) algorithms, and pattern recognition to accurately pinpoint PII ensuring compliance with privacy regulations and minimizing the risk of data breaches. By implementing PII detection guardrails, organizations can enhance their data protection strategies, foster trust with stakeholders, and uphold the integrity of personal information.

This component currently supports two microservices: an OPEA native (free, local, open-source) microservice and a Prediction Guard (API Key required) microservice. Please choose one of the two microservices for PII detection based on your specific use case. If you wish to run both for experimental or comparison purposes, make sure to modify the port configuration of one service to avoid conflicts, as they are configured to use the same port by default.

### PII Detection Microservice

This service uses a [SpaCy](https://spacy.io/) pipeline that is built by a [Microsoft Presidio](https://microsoft.github.io/presidio/) Transformers Nlp Engine. The pipeline contains standard NLP and regex-based recognizers that detect the entities listed [here](https://microsoft.github.io/presidio/supported_entities/) as well as a BERT-based model ([`StanfordAIMI/stanford-deidentifier-base`](https://huggingface.co/StanfordAIMI/stanford-deidentifier-base)) that additionally detects the following PII entities:

- Person
- Location
- Organization
- Age
- ID
- Email
- Date/time
- Phone number
- Nationality/religious/political group

The service takes text as input (TextDoc) and returns either the original text (TextDoc) if no PII is detected, or a list of detected entities (PIIResponseDoc) with the detection details including detection score (probability), detection method and start/end string indices of detection.

Stay tuned for the following future work:

- Entity configurability
- PII replacement

### Prediction Guard PII Detection Microservice

[Prediction Guard](https://docs.predictionguard.com) allows you to utilize hosted open access LLMs, LVMs, and embedding functionality with seamlessly integrated safeguards. In addition to providing a scalable access to open models, Prediction Guard allows you to configure factual consistency checks, toxicity filters, PII filters, and prompt injection blocking. Join the [Prediction Guard Discord channel](https://discord.gg/TFHgnhAFKd) and request an API key to get started.

Detecting Personal Identifiable Information (PII) is important in ensuring that users aren't sending out private data to LLMs. This service allows you to configurably:

1. Detect PII
2. Replace PII (with "faked" information)
3. Mask PII (with placeholders)

## Environment Setup

### Clone OPEA GenAIComps and Setup Environment

Clone this repository at your desired location and set an environment variable for easy setup and usage throughout the instructions.

```bash
git clone https://github.com/opea-project/GenAIComps.git

export OPEA_GENAICOMPS_ROOT=$(pwd)/GenAIComps
```

Set the port that this service will use and the component name

```bash
export PII_DETECTION_PORT=9080
export PII_DETECTION_COMPONENT_NAME="OPEA_NATIVE_PII"
```

By default, this microservice uses `OPEA_NATIVE_PII` which uses [`Microsoft Presidio`](https://microsoft.github.io/presidio/) to locally invoke [`StanfordAIMI/stanford-deidentifier-base`](https://huggingface.co/StanfordAIMI/stanford-deidentifier-base) within a Transformers-based AnalyzerEngine.

#### Alternatively, if you are using Prediction Guard, set the following component name and Prediction Guard API Key:

```bash
export PII_DETECTION_COMPONENT_NAME="PREDICTIONGUARD_PII_DETECTION"
export PREDICTIONGUARD_API_KEY=${your_predictionguard_api_key}
```

## 🚀1. Start Microservice with Python（Option 1）

### 1.1 Install Requirements

```bash
cd $OPEA_GENAICOMPS_ROOT/comps/guardrails/src/pii_detection
pip install -r requirements.txt
```

### 1.2 Start PII Detection Microservice with Python Script

```bash
python opea_pii_detection_microservice.py
```

## 🚀2. Start Microservice with Docker (Option 2)

### For native OPEA Microservice

#### 2.1 Build Docker Image

```bash
cd $OPEA_GENAICOMPS_ROOT
docker build \
  --build-arg https_proxy=$https_proxy \
  --build-arg http_proxy=$http_proxy \
  -t opea/guardrails-pii-detection:latest \
  -f comps/guardrails/src/pii_detection/Dockerfile .
```

#### 2.2.a Run Docker with Compose (Option A)

```bash
cd $OPEA_GENAICOMPS_ROOT/comps/guardrails/deployment/docker_compose
docker compose up -d guardrails-pii-detection-server
```

#### 2.2.b Run Docker with CLI (Option B)

```bash
docker run -d --rm \
    --name="guardrails-pii-detection-server" \
    -p ${PII_DETECTION_PORT}:9080 \
    --ipc=host \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e no_proxy=${no_proxy} \
     opea/guardrails-toxicity-detection:latest
```

### For Prediction Guard Microservice

#### 2.1 Build Docker Image

```bash
cd $OPEA_GENAICOMPS_ROOT
docker build \
  --build-arg https_proxy=$https_proxy \
  --build-arg http_proxy=$http_proxy \
  -t opea/guardrails-pii-predictionguard:latest \
  -f comps/guardrails/src/pii_detection/Dockerfile .
```

#### 2.2.a Run Docker with Compose (Option A)

```bash
cd $OPEA_GENAICOMPS_ROOT/comps/guardrails/deployment/docker_compose
docker compose up -d pii-predictionguard-server
```

#### 2.2.b Run Docker with CLI (Option B)

```bash
docker run -d \
  --name="pii-predictionguard-server" \
  -p ${PII_DETECTION_PORT}:9080 \
  -e PREDICTIONGUARD_API_KEY=$PREDICTIONGUARD_API_KEY \
  -e PII_DETECTION_COMPONENT_NAME: ${PREDICTIONGUARD_PII_DETECTION}
  opea/guardrails-pii-predictionguard:latest
```

## 🚀3. Get Status of Microservice

### For native OPEA Microservice

```bash
docker container logs -f guardrails-pii-detection-server
```

### For Prediction Guard Microservice

```bash
docker container logs -f  pii-predictionguard-server
```

## 🚀4. Consume Microservice

Once microservice starts, users can use examples (bash or python) below to apply PII detection

### For native OPEA Microservice

**Bash Example**:

```bash
curl localhost:${PII_DETECTION_PORT}/v1/pii \
    -X POST \
    -d '{"text":"My name is John Doe and my phone number is (555) 555-5555."}' \
    -H 'Content-Type: application/json'
```

**Python Example:**

```python
import requests
import json
import os

pii_detection_port = os.getenv("PII_DETECTION_PORT")
proxies = {"http": ""}
url = f"http://localhost:{pii_detection_port}/v1/pii"
data = {"text": "My name is John Doe and my phone number is (555) 555-5555."}


try:
    resp = requests.post(url=url, data=data, proxies=proxies)
    print(json.dumps(json.loads(resp.text), indent=4))
    resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
    print("Request successful!")
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
```

**Example Output**:

```json
{
  "id": "4631406f5f91728e45ad27eba062bb4b",
  "detected_pii": [
    {
      "entity_type": "PHONE_NUMBER",
      "start": 44,
      "end": 58,
      "score": 0.9992861151695251,
      "analysis_explanation": null,
      "recognition_metadata": {
        "recognizer_name": "TransformersRecognizer",
        "recognizer_identifier": "TransformersRecognizer_140427422846672"
      }
    },
    {
      "entity_type": "PERSON",
      "start": 12,
      "end": 20,
      "score": 0.8511614799499512,
      "analysis_explanation": null,
      "recognition_metadata": {
        "recognizer_name": "TransformersRecognizer",
        "recognizer_identifier": "TransformersRecognizer_140427422846672"
      }
    }
  ],
  "new_prompt": null
}
```

### For Prediction Guard Microservice

```bash
curl -X POST http://localhost:${PII_DETECTION_PORT}/v1/pii \
    -H 'Content-Type: application/json' \
    -d '{
      "prompt": "My name is John Doe and my phone number is (555) 555-5555.",
      "replace": true,
      "replace_method": "random"
      }'
```

API parameters for Prediction Guard microservice:

- `prompt` (string, required): The text in which you want to detect PII (typically the prompt that you anticipate sending to an LLM)
- `replace` (boolean, optional, default is `false`): `true` if you want to replace the detected PII in the `prompt`
- `replace_method` (string, optional, default is `random`): The method you want to use to replace PII (set to either `random`, `fake`, `category`, `mask`)
