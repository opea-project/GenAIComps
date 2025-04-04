# Docker Compose for LLM Router Microservice

This folder contains a `compose.yaml` file that spins up the **LLM Router** microservice using Docker Compose.

## Introduction

Here, we follow the **GenAIComps** pattern: a **Docker Compose** file defines how to launch our **Router** container with the necessary environment variables and config file.

## What Is Deployed?

- **`router_service`**: A container running the Router microservice.  
  - Exposes port **6000**  
  - Loads its main `config.yaml` from a volume (this is your global config controlling which model endpoints are “weak” vs. “strong,” etc.).  
  - Looks for `HF_TOKEN` (Hugging Face) and `OPENAI_API_KEY` environment variables.

## Files

- **`compose.yaml`**  
  Describes the service, environment variables, ports, etc.

## How To Use

1. **Set Up Environment Variables**  
   - `HF_TOKEN` (if using Hugging Face embeddings or RouteLLM-based code)  

   ```bash
   export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXX
   ```

2. (Optional) Provide a Custom config.yaml

If you want to customize your routes or change controller_config_path, place a config.yaml in this folder (or a known path) with your desired contents.

3. Run Docker Compose