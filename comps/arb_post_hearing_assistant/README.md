# Arbitratory Post-Hearing Assistant Microservice

The **Arbitrator Post-Hearing Assistant** microservice leverages **LangChain** to provide advanced **entity extraction** from arbitration hearing transcripts sourced from multiple platforms, including **Zoom, Slack, Teams**, and other collaboration tools.

## Features

- Arbitration hearing transcripts sourced from multiple platforms, including Zoom, Slack, Teams, and other collaboration tools. 
- Perform **LLM inference** using **Text Generation Inference (TGI)** on **Intel Xeon processors**.
- Generate **summaries** of hearings.
- Extract **metadata** such as participants, case number, next scheduling date, outcome,timestamps, key topics, and actions.
- Backend configurable to use either **[TGI](../../../third_parties/tgi)** or **[vLLM](../../../third_parties/vllm)** for LLM processing.
- Automatically provide **structured insights** ready for downstream applications like case management, reporting, and legal analytics.


## Benefits

- Streamlines post-hearing processing.
- Provides actionable insights quickly and efficiently.
- Integrates seamlessly with other OPEA components.


## Table of Contents

1. [Start Microservice](#start-microservice)
2. [Consume Microservice](#consume-microservice)
---

## Start Microservice

### Set Environment Variables

```bash
export host_ip=${your_host_ip}
export LLM_ENDPOINT_PORT=8008
export OPEA_ARB_POSTHEARING_ASSISTANT_PORT=9000
export HF_TOKEN=${your_hf_api_token}
export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
export LLM_MODEL_ID=${your_hf_llm_model}
export MAX_INPUT_TOKENS=2048
export MAX_TOTAL_TOKENS=4096
```

> MAX_TOTAL_TOKENS must be greater than MAX_INPUT_TOKENS + max_new_tokens + 50 (50 tokens reserved for prompt length).

### Build Docker Images

#### Build Backend LLM Image

For vLLM, refer to [vLLM Build Instructions](../../../third_parties/vllm/).

TGI does not require additional setup.

#### Build Arbitratory Post-Hearing Assistant Microservice Image

```bash
cd ../../../../
docker build -t opea/arb-post-hearing-assistant:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/arb_post_hearing_assistant/src/Dockerfile .
```

### Run Docker Service

You can start the service using either the CLI or Docker Compose.

#### Option A: Run with Docker CLI

1. Start the backend LLM service ([TGI](../../../third_parties/tgi) or [vLLM](../../../third_parties/vllm)).

2. Start Arbitratory Post-Hearing Assistant microservice:

```bash
export OPEA_ARB_POSTHEARING_ASSISTANT_COMPONENT_NAME="OpeaArbPostHearingAssistantTgi" # or "OpeaArbPostHearingAssistantvllm"
docker run -d \
    --name="arb-post-hearing-assistant-server" \
    -p 9000:9000 \
    --ipc=host \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e LLM_MODEL_ID=$LLM_MODEL_ID \
    -e LLM_ENDPOINT=$LLM_ENDPOINT \
    -e HF_TOKEN=$HF_TOKEN \
    -e OPEA_ARB_POSTHEARING_ASSISTANT_COMPONENT_NAME=$OPEA_ARB_POSTHEARING_ASSISTANT_COMPONENT_NAME \
    -e MAX_INPUT_TOKENS=$MAX_INPUT_TOKENS \
    -e MAX_TOTAL_TOKENS=$MAX_TOTAL_TOKENS \
    opea/arb-post-hearing-assistant:latest
```

#### Option B: Run with Docker Compose

```bash
export service_name="arbPostHearingAssistant-tgi"
# Alternatives: , "arbPostHearingAssistant-vllm"

cd ../../deployment/docker_compose/
docker compose -f compose.yaml up ${service_name} -d
```
