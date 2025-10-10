# Arbitrator Post-Hearing Assistant

The **Arbitrator Post-Hearing Assistant** microservice leverages **LangChain** to provide advanced **entity extraction** from arbitration hearing transcripts sourced from multiple platforms, including **Zoom, Slack, Teams**, and other collaboration tools.

## Features

- Arbitration hearing transcripts sourced from multiple platforms, including Zoom, Slack, Teams, and other collaboration tools.
- Perform **LLM inference** using **Text Generation Inference (TGI)** on **Intel Xeon processors**.
- Generate **summaries** of hearings.
- Extract **metadata** such as participants, case number, next scheduling date, outcome,timestamps, key topics, and actions.
- Backend configurable to use either **[TGI](../third_parties/tgi)** or **[vLLM](../third_parties/vllm)** for LLM processing.
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

For vLLM, refer to [vLLM Build Instructions](../third_parties/vllm/).

TGI does not require additional setup.

#### Build Arbitratory Post-Hearing Assistant Microservice Image

```bash
cd ../../../../
docker build -t opea/arb-post-hearing-assistant:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/arb_post_hearing_assistant/src/Dockerfile .
```

### Run Docker Service

You can start the service using either the CLI or Docker Compose.

#### Option A: Run with Docker CLI

1. Start the backend LLM service ([TGI](../third_parties/tgi) or [vLLM](../third_parties/vllm)).

2. Start Arbitratory Post-Hearing Assistant microservice:

```bash
export OPEA_ARB_POSTHEARING_ASSISTANT_COMPONENT_NAME="OpeaArbPostHearingAssistantTgi" # or "OpeaArbPostHearingAssistantVllm"
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
docker compose -f arb_post_hearing_assistant-compose.yaml up ${service_name} -d
```

#### Basic usage

```bash
curl http://0.0.0.0:9000/v1/arb-post-hearing \-X POST \-H 'Content-Type: application/json' \-d '{ "messages": "[10:00 AM] Arbitrator A: Good morning. This hearing is now in session for Case No. ARB/2025/0917. Let’s begin with appearances. [10:01 AM] Advocate B (for Party X): Good morning, Your Honor. I appear for the claimant, Mr. X. [10:01 AM] Advocate C (for Party Y): Good morning. I represent the respondent, Ms. Y. [10:03 AM] Arbitrator A: Thank you. Let’s proceed with Party X’s opening statement. [10:04 AM] Advocate B: Party Y failed to deliver services as per the agreement dated 15 March 2023. We’ve submitted relevant documents including emails and payment records. The delay caused significant financial loss to our client. [10:15 AM] Advocate C: We deny the breach. Delays were due to regulatory hurdles beyond our control. Party X also failed to provide timely approvals, which contributed to the delay. [10:30 AM] Arbitrator A: Let’s focus on Clause Z of the agreement. I’d like both parties to submit written arguments on the applicability of force majeure and the timeline of approvals. [11:00 AM] Advocate B: Understood. We will submit by the deadline. [11:01 AM] Advocate C: Agreed. [11:02 AM] Arbitrator A: Next hearing is scheduled for 10 October 2024 at 10:30 AM. Please ensure your witnesses are available for cross-examination. [4:45 PM] Arbitrator A: This session is adjourned. Thank you, everyone.","max_tokens": 32,"language": "en" }'

```
