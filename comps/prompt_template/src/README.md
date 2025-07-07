# Prompt Template microservice

The Prompt Template microservice dynamically generates system and user prompts based on structured inputs and document context. It supports usage in LLM pipelines to customize prompt formatting with reranked documents, conversation history, and user queries.

## Getting started

### 🚀1. Start Prompt Template Microservice with Python (Option 1)

To start the Prompt Template microservice, you need to install Python packages first.

#### 1.1. Install Requirements

```bash
pip install -r requirements.txt
```

#### 1.2. Start Microservice

```bash
python opea_prompt_template_microservice.py
```

### 🚀2. Start Prompt Template Microservice with Docker (Option 2)

#### 2.1. Build the Docker Image:

Use the below docker build command to create the image:

```bash
cd ../../../
docker build -t opea/prompt-template:latest -f comps/prompt_template/src/Dockerfile .
```

Please note that the building process may take a while to complete.

#### 2.2. Run the Docker Container:

```bash
docker run -d --name="prompt-template-microservice" \
  -p 7900:7900 \
  --net=host \
  --ipc=host \
  opea/prompt-template:latest
```

### 3. Verify the Prompt Template Microservice

#### 3.1. Check Status

```bash
curl http://localhost:7900/v1/health_check \
  -X GET \
  -H 'Content-Type: application/json'
```

#### 3.2. Sending a Request

##### 3.2.1 Default Template Generation

Generates the prompt using the default template:

**Example Input**

```bash
curl -X POST -H "Content-Type: application/json" -d @- http://localhost:7900/v1/prompt_template <<JSON_DATA
{
  "data": {
    "user_prompt": "What is Deep Learning?",
    "reranked_docs": [{ "text": "Deep Learning is a subfield of machine learning..." }]
  },
  "conversation_history": [
    { "question": "Hello", "answer": "Hello as well" },
    { "question": "How are you?", "answer": "I am good, thank you!" }
  ],
  "system_prompt_template": "",
  "user_prompt_template": ""
}
JSON_DATA
```

**Example Output**

A chat_template starting with the default assistant description.

```json
{
  "id": "4e799abdf5f09433adc276b511a8b0ae",
  "model": null,
  "query": "What is Deep Learning?",
  "max_tokens": 1024,
  "max_new_tokens": 1024,
  "top_k": 10,
  "top_p": 0.95,
  "typical_p": 0.95,
  "temperature": 0.01,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "repetition_penalty": 1.03,
  "stream": true,
  "language": "auto",
  "input_guardrail_params": null,
  "output_guardrail_params": null,
  "chat_template": "### You are a helpful, respectful, and honest assistant to help the user with questions. Please refer to the search results obtained from the local knowledge base. Refer also to the conversation history if you think it is relevant to the current question. Ignore all information that you think is not relevant to the question. If you don'\''t know the answer to a question, please don'\''t share false information. \n ### Search results: [File: Unknown Source]\nDeep Learning is...\n### Conversation history: User: Hello\nAssistant: Hello as well\nUser: How are you?\nAssistant: I am good, thank you!\nUser: Who are you?\nAssistant: I am a robot\n### Question: What is Deep Learning? \n\n### Answer:",
  "documents": []
}
```

##### 3.2.2 Custom Prompt Template

You can provide custom system and user prompt templates:

**Example Input**

```bash
curl -X POST -H "Content-Type: application/json" -d @- http://localhost:7900/v1/prompt_template <<JSON_DATA
{
  "data": {
    "initial_query": "What is Deep Learning?",
    "reranked_docs": [{ "text": "Deep Learning is..." }]
  },
  "system_prompt_template": "### Please refer to the search results obtained from the local knowledge base. But be careful to not incorporate information that you think is not relevant to the question. If you don't know the answer to a question, please don't share false information. ### Search results: {reranked_docs}",
  "user_prompt_template": "### Question: {initial_query} \\n### Answer:"
}
JSON_DATA
```

**Example Output**

Custom instructions about using search results in the chat_template.

```json
{
  "id": "b1f1cec396954d5dc1b942f5959d556d",
  "model": null,
  "query": "What is Deep Learning?",
  "max_tokens": 1024,
  "max_new_tokens": 1024,
  "top_k": 10,
  "top_p": 0.95,
  "typical_p": 0.95,
  "temperature": 0.01,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "repetition_penalty": 1.03,
  "stream": true,
  "language": "auto",
  "input_guardrail_params": null,
  "output_guardrail_params": null,
  "chat_template": "### Please refer to the search results obtained from the local knowledge base. But be careful to not incorporate information that you think is not relevant to the question. If you don'\''t know the answer to a question, please don'\''t share false information. ### Search results: [File: Unknown Source]\nDeep Learning is...\n### Question: What is Deep Learning? \n### Answer:",
  "documents": []
}
```

##### 3.2.3 Translation Scenario

Using a translation-related prompt template:

**Example Input**

```bash
curl -X POST -H "Content-Type: application/json" -d @- http://localhost:7900/v1/prompt_template <<JSON_DATA
{
  "data": {
    "initial_query": "What is Deep Learning?",
    "source_lang": "chinese",
    "target_lang": "english"
  },
  "system_prompt_template": "### You are a helpful, respectful, and honest assistant to help the user with translations. Translate this from {source_lang} to {target_lang}.",
  "user_prompt_template": "### Question: {initial_query} \\n### Answer:"
}
JSON_DATA
```

**Example Output**

A translation instruction like: Translate this from chinese to english.

```json
{
  "id": "4f5e0024c2330a7be065b370d02e061f",
  "model": null,
  "query": "什么是深度学习？",
  "max_tokens": 1024,
  "max_new_tokens": 1024,
  "top_k": 10,
  "top_p": 0.95,
  "typical_p": 0.95,
  "temperature": 0.01,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "repetition_penalty": 1.03,
  "stream": true,
  "language": "auto",
  "input_guardrail_params": null,
  "output_guardrail_params": null,
  "chat_template": "### You are a helpful, respectful, and honest assistant to help the user with translations. Translate this from chinese to english.\n### Question: 什么是深度学习？ \n### Answer:",
  "documents": []
}
```
