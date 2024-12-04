# Hallucination Detection Microservice

## Introduction

Hallucination in AI, particularyly in large language models (LLMs), spans a wide range of issues that can impact reliability, trustworthiness, and utility of AI-generated content. The content could be plausible-sounding and factually incorrect, irrelevant, or entirely fabricated. This phenonenon occurs when the model generates outputs that are not grounded in the input context, training data, or real-world knowledge. While LLMs excel at generating coherent response, hallucinations pose a critical challenge for applications that demand accuracy, reliability, and trustworthiness.

### Forms of Hallucination

* **Factual Errors**: The AI generates responses containing incorrect or fabricated facts. _Example_: Claiming a historical event occured when it did not. 

* **Logical Inconsistencies**: Ouputs that fail to follow logical reasoning or contradict themselves. _Example_: Stating that a person is alive in one sentence and deceased in another.

* **Context Misalignment**: Responses that diverge from the input prompt or fail to address the intended context. _Example_: Providing irrelevant information or deviating from topic. 

* **Fabricated References**: Creating citations, statistics, or other details that appear authentic but lack real-world grounding. _Example_: Inventing a study or paper that doesn't exist. 

### Importance of Hallucination Detection
The Importance of hallucination detection cannot be overstated. Ensuring the factual correctness and contextual fidelity of AI-generated content is essential for: 

* __Building Trust__: Reducing hallucinations foster user confidence in AI system. 
* __Ensuring Compliance__: Meeting legal and ethical standards in regulated industries. 
* __Enhancing Reliability__: Improving the overall robustness and performance of AI applications.  


### Define the Scope of Our Hallucination Detection
Tackling the entire scope of hallucination is beyond our immediate scope. Training datasets inherently lag behind the question-and-answer needs due to their static nature. Also, Retrieval-Augmented Generation (RAG) is emerging as a preferred approach for LLMs, where model ouputs are grounded in retrieved context to enhance accuracy and relevance and rely on integration of Document-Question-Answer triplets. 

Therefore, we focus on detecting contextualized hallucinations with the following strategies: 
* Using LLM-as-a-judge to evaluate hallucinations.
* Detect whether Context-Question-Answer triplet contains hallucinations.





## 🚀1. Start Microservice based on vLLM endpoint on Intel Gaudi Accelerator
### 1.1 Define Environment Variables
```bash
export HUGGINGFACEHUB_API_TOKEN=<token>
export vLLM_ENDPOINT="http://${your_ip}:9008"
export LLM_MODEL="PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct"
```
For gated models such as `LLAMA-2`, you will have to pass the environment HUGGINGFACEHUB_API_TOKEN. Please follow this link [huggingface token](https://huggingface.co/docs/hub/security-tokens) to get the access token and export `HUGGINGFACEHUB_API_TOKEN` environment with the token.




### 1.2 Configure vLLM Service on Gaudi Accelerator
#### build docker
```bash
bash ./build_docker_vllm.sh hpu
```
### 1.3 Launch vLLM Service
#### Launch vLLM service on a single node
```bash

bash ./launch_vllm_service.sh ${port_number} ${model_name} hpu 1
```

## 2. Set up Hallucination Microservice

Then we wrap the vLLM Service into Hallucination Microservice. 

### 2.1 Build Docker
```bash
bash build_docker_microservice.sh
```

### 2.2 Launch Hallucination Microservice
```bash
bash launch_microservice.sh
```


## 🚀3. Get Status of Hallucination Microservice

```bash
docker container logs -f guardrails-hallucination-detection
```
## 🚀4. Consume Guardrail Micorservice Post-LLM

Once microservice starts, users can use examples (bash or python) below to apply hallucination detection for LLM's response (Post-LLM)

**Bash:**

```bash
curl localhost:9092/v1/bias
    -X POST
    -d '{"text":"John McCain exposed as an unprincipled politician"}'
    -H 'Content-Type: application/json'
```

Example Output:

```bash
"\nI'm sorry, but your query or LLM's response is BIASED with an score of 0.74 (0-1)!!!\n"
```

**Python Script:**

```python
import requests
import json

proxies = {"http": ""}
url = "http://localhost:9092/v1/bias"
data = {"text": "John McCain exposed as an unprincipled politician"}


try:
    resp = requests.post(url=url, data=data, proxies=proxies)
    print(resp.text)
    resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
    print("Request successful!")
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
```

```bash
pip install -r requirements.txt
```




