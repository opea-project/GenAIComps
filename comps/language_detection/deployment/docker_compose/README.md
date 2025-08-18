# Deploying language-detection Service

This document provides a comprehensive guide to deploying the language-detection microservice pipeline on Intel platforms.

This guide covers two deployment methods:

- [🚀 1. Quick Start with Docker Compose](#-1-quick-start-with-docker-compose): The recommended method for a fast and easy setup.
- [🚀 2. Manual Step-by-Step Deployment (Advanced)](#-2-manual-step-by-step-deployment-advanced): For users who want to build and run each container individually.
- [🚀 3. Start Microservice with Python](#-3-start-microservice-with-python): For users who prefer to run the ASR microservice directly with Python scripts.

## 🚀 1. Quick Start with Docker Compose

This method uses Docker Compose to start all necessary services with a single command. It is the fastest and easiest way to get the service running.

### 1.1. Access the Code

Clone the repository and navigate to the deployment directory:

```bash
git clone https://github.com/opea-project/GenAIComps.git
cd GenAIComps/comps/language_detection/deployment/docker_compose
```

### 1.2. Deploy the Service

Choose the command corresponding to your target platform.

```bash
docker compose -f compose.yaml up language-detection -d
```

### 1.3. Validate the Service

#### 1.3.1 Pipeline Mode

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

#### 1.3.2 Standalone Mode

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

### 1.4. Clean Up the Deployment

To stop and remove the containers, run the following command from the `comps/language_detection/deployment/docker_compose` directory:

```bash
docker compose down
```

---

## 🚀 2. Manual Step-by-Step Deployment (Advanced)

This section provides detailed instructions for building the Docker images and running each microservice container individually.

### 2.1. Clone the Repository

If you haven't already, clone the repository and navigate to the root directory:

```bash
git clone https://github.com/opea-project/GenAIComps.git
cd GenAIComps
```

### 2.2. Build the Docker Images

Use the below docker build command to create the image:

```bash
docker build -t opea/language-detection:latest -f comps/language_detection/src/Dockerfile .
```

Please note that the building process may take a while to complete.

### 2.3. Configuration Options

The configuration for the Language Detection Microservice can be adjusted by exporting environment variable.

| Environment Variable     | Description                            |
| ------------------------ | -------------------------------------- |
| `LANG_DETECT_STANDALONE` | Set this to `True` for Standalone mode |

### 2.4. Run the Microservice Containers

```bash
docker run -d --name="language-detection-microservice" \
  -p 8069:8069\
  --net=host \
  --ipc=host \
  opea/language-detection:latest
```

### 2.5. Validate the Service

Reference to（[validate the service](#13-validate-the-service)）

### 2.6. Clean Up the Deployment

To stop and remove the containers you started manually, use the `docker stop` and `docker rm` commands.

```bash
docker stop language-detection
docker rm language-detection
```

---

## 🚀 3. Start Language Detection Microservice with Python

To start the Language Detection microservice, you need to install python packages first.

### 3.1. Install Requirements

```bash
pip install -r requirements.txt
```

### 3.2. Start Microservice

```bash
python opea_language_detection_microservice.py
```
