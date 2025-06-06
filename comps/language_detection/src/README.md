# Language Detection microservice

The Language Detection microservice can be run in 2 modes:

1. Pipeline: This mode adds multilingual support to ChatQnA pipelines. The microservice detects the language of the user's query as well as the LLM generated response to set up a prompt for translation.

2. Standalone: This mode supports standalone translation. The microservice detects the language of the provided text. It then sets up a prompt for translating the provided text from the source language (detected language) to the provided target language.

## Configuration Options

The configuration for the Language Detection Microservice can be adjusted by exporting environment variable.

| Environment Variable     | Description                            |
| ------------------------ | -------------------------------------- |
| `LANG_DETECT_STANDALONE` | Set this to `True` for Standalone mode |

## Getting started

### 🚀1. Start Language Detection Microservice with Python (Option 1)

To start the Language Detection microservice, you need to install python packages first.

#### 1.1. Install Requirements

```bash
pip install -r requirements.txt
```

#### 1.2. Start Microservice

```bash
python opea_language_detection_microservice.py
```

### 🚀2. Start Language Detection Microservice with Docker (Option 2)

#### 2.1. Build the Docker Image:

Use the below docker build command to create the image:

```bash
cd ../../../
docker build -t opea/language-detection:latest -f comps/language_detection/src/Dockerfile .
```

Please note that the building process may take a while to complete.

#### 2.2. Run the Docker Container:

```bash
docker run -d --name="language-detection-microservice" \
  -p 8069:8069\
  --net=host \
  --ipc=host \
  opea/language-detection:latest
```

### 3. Verify the Language Detection Microservice

#### 3.1. Check Status

```bash
curl http://localhost:8069/v1/health_check \
  -X GET \
  -H 'Content-Type: application/json'
```

#### 3.2. Sending a Request

##### 3.2.1 Pipeline Mode

The input request consists of the answer that has to be translated and a prompt containing the user's query.

**Example Input**

```bash
curl -X POST -H "Content-Type: application/json" -d @- http://localhost:8069/v1/language_detection <<JSON_DATA
{
  "text": "Hi. I am doing fine.",
  "prompt": "### You are a helpful, respectful, and honest assistant to help the user with questions. \
Please refer to the search results obtained from the local knowledge base. \
But be careful to not incorporate information that you think is not relevant to the question. \
If you don't know the answer to a question, please don't share false information. \
### Search results:   \n
### Question: 你好。你好吗？ \n
### Answer:"
}
JSON_DATA
```

**Example Output**

The output contains the answer, prompt template, source language and target language.

```json
{
  "id": "1b16e065a1fcbdb4d999fd3d09a619cb",
  "data": { "text": "Hi. I am doing fine.", "source_lang": "English", "target_lang": "Chinese" },
  "prompt_template": "\n Translate this from {source_lang} to {target_lang}:\n   {source_lang}:\n   {text}\n\n  {target_lang}: \n "
}
```

##### 3.2.2 Standalone Mode

The input request consists of the text that has to be translated and the target language.

**Example Input**

```bash
curl -X POST -H "Content-Type: application/json" -d @- http://localhost:8069/v1/language_detection <<JSON_DATA
{
  "text": "Hi. I am doing fine.",
  "target_language": "Chinese"
}
JSON_DATA
```

**Example Output**

The output contains the original text, prompt template, source language and target language.

```json
{
  "id": "1b16e065a1fcbdb4d999fd3d09a619cb",
  "data": { "text": "Hi. I am doing fine.", "source_lang": "English", "target_lang": "Chinese" },
  "prompt_template": "\n Translate this from {source_lang} to {target_lang}:\n   {source_lang}:\n   {text}\n\n  {target_lang}: \n "
}
```
