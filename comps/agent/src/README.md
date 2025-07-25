# Agent Microservice

This agent microservice is built on Langchain/Langgraph frameworks. Agents integrate the reasoning capabilities of large language models (LLMs) with the ability to take actionable steps, creating a more sophisticated system that can understand and process information, evaluate situations, take appropriate actions, communicate responses, and track ongoing situations.

## Table of Contents

1. [Quick Start Deployment](#quick-start-deployment): Step-by-step guide to quickly deploy Opea components.
   - [Build Docker Image](#build-docker-image)
   - [Setup Environments](#setup-environments)
   - [Start Agent Microservices](#start-agent-microservices)
   - [Validate Microservice](#validate-microservice)
2. [Supported Agent Types](#supported-agent-types): Overview of all agent types currently supported.
3. [Docker Compose Files](#docker-compose-files): Standard Docker Compose configurations for easy setup.
4. [Other Supported Features](#other-supported-features): Includes integrated tools, OpenAI-compatible APIs, and agent memory support.
   - [Tools](#tools)
   - [Agent APIs](#agent-apis)
   - [Agent memory](#agent-memory)
   - [Run LLMs from OpenAI](#run-llms-from-openai)
   - [Run LLMs with OpenAI-compatible APIs on Remote Servers](#run-llms-with-openai-compatible-apis-on-remote-servers)
5. [Customizations](#customizations): Guide to customizing tools and agent strategies for specific needs.
   - [Customize tools](#customize-tools)
   - [Customize agent strategy](#customize-agent-strategy)

## Quick Start Deployment

### Build Docker Image

```bash
cd GenAIComps/ # back to GenAIComps/ folder
docker build -t opea/agent:latest -f comps/agent/src/Dockerfile . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
```

### Setup Environments

```bash
export ip_address=$(hostname -I | awk '{print $1}')
export model="meta-llama/Meta-Llama-3.1-70B-Instruct"
export HF_TOKEN=${HF_TOKEN}
export vllm_volume=${YOUR_LOCAL_DIR_FOR_MODELS}
```

### Start Agent Microservices

```bash
# build vLLM image
git clone https://github.com/HabanaAI/vllm-fork.git
cd ./vllm-fork
VLLM_VER=v0.6.6.post1+Gaudi-1.20.0
git checkout ${VLLM_VER} &> /dev/null
docker build -f Dockerfile.hpu -t opea/vllm-gaudi:latest --shm-size=128g . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy

# serve vllm on 4 Gaudi2 cards
docker run -d --runtime=habana --rm --name "comps-vllm-gaudi-service" -p 8080:8000 -v $vllm_volume:/data -e HF_TOKEN=$HF_TOKEN -e HF_HOME=/data -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e VLLM_SKIP_WARMUP=true --cap-add=sys_nice --ipc=host opea/vllm-gaudi:latest --model ${model} --max-seq-len-to-capture 16384 --enable-auto-tool-choice --tool-call-parser llama3_json --guided-decoding-backend lm-format-enforcer --tensor-parallel-size 4

# check status
docker logs comps-vllm-gaudi-service

# Agent
docker run -d --runtime=runc --name="comps-agent-endpoint" -v $WORKPATH/comps/agent/src/tools:/home/user/comps/agent/src/tools -p 9090:9090 --ipc=host -e HF_TOKEN=${HF_TOKEN} -e model=${model} -e ip_address=${ip_address} -e strategy=react_llama -e with_memory=true -e llm_endpoint_url=http://${ip_address}:8080 -e llm_engine=vllm -e recursion_limit=15 -e require_human_feedback=false -e tools=/home/user/comps/agent/src/tools/custom_tools.yaml opea/agent:latest

# check status
docker logs comps-agent-endpoint
```

To run the agent microservice in debug mode, run the command below:

```bash
docker run --rm --runtime=runc --name="comps-agent-endpoint" -v ./comps/agent/src/:/home/user/comps/agent/src/ -p 9090:9090 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e HF_TOKEN=${HF_TOKEN} -e model=${model} -e ip_address=${ip_address} -e strategy=react_llama -e with_memory=true -e llm_endpoint_url=http://${ip_address}:8080 -e llm_engine=vllm -e recursion_limit=15 -e require_human_feedback=false -e tools=/home/user/comps/agent/src/tools/custom_tools.yaml opea/agent:latest
```

### Validate Microservice

Once microservice starts, user can use below script to invoke.

#### Use chat completions API

For multi-turn conversations, first specify a `thread_id`.

```bash
export thread_id=<thread-id>
curl http://${ip_address}:9090/v1/chat/completions -X POST -H "Content-Type: application/json" -d '{
     "messages": "What is OPEA project?",
     "thread_id":${thread_id},
     "stream":true
    }'

# expected output
data: 'The OPEA project is .....</s>' # just showing partial example here.
data: [DONE]
```

#### Use assistants APIs

```bash
# step1 create assistant to get `asssistant_id`
curl http://${ip_address}:9090/v1/assistants -X POST -H "Content-Type: application/json" -d '{
     "agent_config": {"llm_engine": "vllm", "llm_endpoint_url": "http://${ip_address}:8080", "tools": "/home/user/comps/agent/src/tools/custom_tools.yaml"}
    }'

## if want to persist your agent messages, set store config like this:
curl http://${ip_address}:9090/v1/assistants -X POST -H "Content-Type: application/json" -d '{
     "agent_config": {"llm_engine": "vllm", "llm_endpoint_url": "http://${ip_address}:8080", "tools": "/home/user/comps/agent/src/tools/custom_tools.yaml","with_memory":true, "memory_type":"store", "store_config":{"redis_uri":"redis://${ip_address}:6379"}}
    }'

# step2 create thread to get `thread_id`
curl http://${ip_address}:9090/v1/threads -X POST -H "Content-Type: application/json" -d '{}'

# step3 create messages
curl http://${ip_address}:9090/v1/threads/{thread_id}/messages -X POST -H "Content-Type: application/json" -d '{"role": "user", "content": "What is OPEA project?"}'


## if agent is set with `memory_type`=store, should add `assistant_id` in the messages for store
curl http://${ip_address}:9090/v1/threads/{thread_id}/messages -X POST -H "Content-Type: application/json" -d '{"role": "user", "content": "What is OPEA project?", "assistant_id": "{assistant_id}"}'

# step4 run
curl http://${ip_address}:9090/v1/threads/{thread_id}/runs -X POST -H "Content-Type: application/json" -d '{"assistant_id": "{assistant_id}"}'
```

## Supported agent types

We currently support the following types of agents. Please refer to the example config yaml (links in the table in [Docker Compose Files](#docker-compose-files)) for each agent strategy to see what environment variables need to be set up.

1. ReAct: use `react_langchain` or `react_langgraph` or `react_llama` as strategy. First introduced in this seminal [paper](https://arxiv.org/abs/2210.03629). The ReAct agent engages in "reason-act-observe" cycles to solve problems. Please refer to this [doc](https://python.langchain.com/v0.2/docs/how_to/migrate_agent/) to understand the differences between the langchain and langgraph versions of react agents. See table below to understand the validated LLMs for each react strategy. We recommend using `react_llama` as it has the most features enabled, including agent memory, multi-turn conversations and assistants APIs.
2. RAG agent: use `rag_agent` or `rag_agent_llama` strategy. This agent is specifically designed for improving RAG performance. It has the capability to rephrase query, check relevancy of retrieved context, and iterate if context is not relevant. See table below to understand the validated LLMs for each rag agent strategy.
3. Plan and execute: `plan_execute` strategy. This type of agent first makes a step-by-step plan given a user request, and then execute the plan sequentially (or in parallel, to be implemented in future). If the execution results can solve the problem, then the agent will output an answer; otherwise, it will replan and execute again.
4. SQL agent: use `sql_agent_llama` or `sql_agent` strategy. This agent is specifically designed and optimized for answering questions aabout data in SQL databases. Users need to specify `db_name` and `db_path` for the agent to access the SQL database. For more technical details read descriptions [here](integrations/strategy/sqlagent/README.md).

**Note**:

1. Due to the limitations in support for tool calling by TGI and vllm, we have developed subcategories of agent strategies (`rag_agent_llama`, `react_llama` and `sql_agent_llama`) specifically designed for open-source LLMs served with TGI and vllm.
2. Currently only `react_llama` agent supports memory and multi-turn conversations.
3. For advanced developers who want to implement their own agent strategies, please refer to [Customize agent strategy](#customize-agent-strategy) below.

## Docker Compose Files

We provide multiple Docker Compose YAML files to facilitate deployment of different agent configurations and LLM serving options. Each YAML file demonstrates how to allocate resources and set environment variables for specific agent strategies and LLM engines, enabling flexible and efficient deployment on Intel® Gaudi® or other platforms.

- Each compose file corresponds to a distinct agent type or strategy, such as ReAct (with LangChain, LangGraph, or Llama backend), RAG agent, or Plan & Execute agent.

- Example YAMLs are pre-configured to use either open-source LLMs (via vllm) or OpenAI APIs, with recommended hardware bindings and agent parameters.

- Users can select and launch the desired agent setup by choosing the appropriate YAML file (e.g., react_langchain.yaml, rag_agent.yaml), ensuring optimal compatibility and performance for their use case.

- The provided compose files illustrate best practices for managing device allocation, parallelism, and LLM-specific options, simplifying multi-agent and multi-model deployments.

| Agent type       | `strategy` arg    | Validated LLMs (serving SW)                                                                                                                                                       | Notes                                                                                                                                                                                                                                                                          | Example config yaml                                               |
| ---------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| ReAct            | `react_langchain` | [llama3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) (vllm-gaudi)                                                                                    | Only allows tools with one input variable                                                                                                                                                                                                                                      | [react_langchain yaml](../../../tests/agent/react_langchain.yaml) |
| ReAct            | `react_langgraph` | GPT-4o-mini, [llama3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) (vllm-gaudi)                                                                       | if using vllm, need to specify `--enable-auto-tool-choice --tool-call-parser ${model_parser}`, refer to vllm docs for more info, only one tool call in each LLM output due to the limitations of llama3.1 modal and vllm tool call parser.                                     | [react_langgraph yaml](../../../tests/agent/react_vllm.yaml)      |
| ReAct            | `react_llama`     | [llama3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), [llama3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)(vllm-gaudi)  | Recommended for open-source LLMs, supports multiple tools and parallel tool calls.                                                                                                                                                                                             | [react_llama yaml](../../../tests/agent/reactllama.yaml)          |
| RAG agent        | `rag_agent`       | GPT-4o-mini                                                                                                                                                                       |                                                                                                                                                                                                                                                                                | [rag_agent yaml](../../../tests/agent/ragagent_openai.yaml)       |
| RAG agent        | `rag_agent_llama` | [llama3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), [llama3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) (vllm-gaudi) | Recommended for open-source LLMs, only allows 1 tool with input variable to be "query"                                                                                                                                                                                         | [rag_agent_llama yaml](../../../tests/agent/ragagent.yaml)        |
| Plan and execute | `plan_execute`    | GPT-4o-mini, [llama3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) (vllm-gaudi)                                                                       | use `--guided-decoding-backend lm-format-enforcer` when launching vllm.                                                                                                                                                                                                        | [plan_execute yaml](../../../tests/agent/planexec_openai.yaml)    |
| SQL agent        | `sql_agent_llama` | [llama3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), [llama3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) (vllm-gaudi) | database query tool is natively integrated using Langchain's [QuerySQLDataBaseTool](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.sql_database.tool.QuerySQLDatabaseTool.html). User can also register their own tools with this agent. | [sql_agent_llama yaml](../../../tests/agent/sql_agent_llama.yaml) |
| SQL agent        | `sql_agent`       | GPT-4o-mini                                                                                                                                                                       | database query tool is natively integrated using Langchain's [QuerySQLDataBaseTool](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.sql_database.tool.QuerySQLDatabaseTool.html). User can also register their own tools with this agent. | [sql_agent yaml](../../../tests/agent/sql_agent_openai.yaml)      |

## Other Supported Features

### Tools

The tools are registered with a yaml file. We support the following types of tools:

1. Endpoint: user to provide url
2. User-defined python functions. This is usually used to wrap endpoints with request post or simple pre/post-processing.
3. Langchain tool modules.

Examples of how to register tools can be found in [Customize tools](#customize-tools) below.

### Agent APIs

We support two sets of APIs that are OpenAI compatible:

1. OpenAI compatible chat completions API. Example usage with Python code below.

   ```python
   url = f"http://{ip_address}:{agent_port}/v1/chat/completions"

   # single-turn, not streaming -> if agent is used as a worker agent (i.e., tool for supervisor agent)
   payload = {"messages": query, "stream": false}
   resp = requests.post(url=url, json=payload, proxies=proxies, stream=False)

   # multi-turn, streaming -> to interface with users
   query = {"role": "user", "messages": user_message, "thread_id": thread_id, "stream": stream}
   content = json.dumps(query)
   resp = requests.post(url=url, data=content, proxies=proxies, stream=True)
   for line in resp.iter_lines(decode_unicode=True):
       print(line)
   ```

2. OpenAI compatible assistants APIs.

   See example Python code [here](./test_assistant_api.py). There are 4 steps:

   Step 1. create an assistant: /v1/assistants

   Step 2. create a thread: /v1/threads

   Step 3. send a message to the thread: /v1/threads/{thread_id}/messages

   Step 4. run the assistant: /v1/threads/{thread_id}/runs

**Note**:

1. Currently only `react_llama` agent is enabled for assistants APIs.
2. Not all keywords of OpenAI APIs are supported yet.

### Agent memory

We currently supports two types of memory.

1. `checkpointer`: agent memory stored in RAM, so is volatile, the memory contains agent states within a thread. Used to enable multi-turn conversations between the user and the agent. Both chat completions API and assistants APIs support this type of memory.
2. `store`: agent memory stored in Redis database, contains agent states in all threads. Only assistants APIs support this type of memory. Used to enable multi-turn conversations. In future we will explore algorithms to take advantage of the info contained in previous conversations to improve agent's performance.

**Note**: Currently only `react_llama` agent supports memory and multi-turn conversations.

#### How to enable agent memory?

Specify `with_memory`=True. If want to use persistent memory, specify `memory_type`=`store`, and you need to launch a Redis database using the command below.

```bash
# you can change the port from 6379 to another one.
docker run -d -it -p 6379:6379 --rm --name "test-persistent-redis" --net=host --ipc=host --name redis-vector-db redis/redis-stack:7.2.0-v9
```

Examples of python code for multi-turn conversations using agent memory:

1. [chat completions API with checkpointer](./test_chat_completion_multiturn.py)
2. [assistants APIs with persistent store memory](./test_assistant_api.py)

To run the two examples above, first launch the agent microservice using [this docker compose yaml](../../../tests/agent/reactllama.yaml).

### Run LLMs from OpenAI

To run any model from OpenAI, just specify the environment variable `OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY=<openai-api-key>
```

These also need to be passed in to the `docker run` command, or included in a YAML file when running `docker compose`.

### Run LLMs with OpenAI-compatible APIs on Remote Servers

To run the text generation portion using LLMs deployed on a remote server, specify the following environment variables:

```bash
export api_key=<openai-api-key>
export model=<model-card>
export LLM_ENDPOINT_URL=<inference-endpoint>
```

These also need to be passed in to the `docker run` command, or included in a YAML file when running `docker compose`.

#### Notes

- For `LLM_ENDPOINT_URL`, there is no need to include `v1`.

## Customizations

### Customize tools

- Define tools

```bash
mkdir -p my_tools
vim my_tools/custom_tools.yaml

# [tool_name]
#   description: [description of this tool]
#   env: [env variables such as API_TOKEN]
#   pip_dependencies: [pip dependencies, separate by ,]
#   callable_api: [2 options provided - function_call, pre-defined-tools]
#   args_schema:
#     [arg_name]:
#       type: [str, int]
#       description: [description of this argument]
#   return_output: [return output variable name]
```

example - my_tools/custom_tools.yaml

```yaml
# Follow example below to add your tool
opea_index_retriever:
  description: Retrieve related information of Intel OPEA project based on input query.
  callable_api: tools.py:opea_rag_query
  args_schema:
    query:
      type: str
      description: Question query
  return_output: retrieved_data
```

example - my_tools/tools.py

```python
def opea_rag_query(query):
    ip_address = os.environ.get("ip_address")
    url = f"http://{ip_address}:8889/v1/retrievaltool"
    content = json.dumps({"text": query})
    print(url, content)
    try:
        resp = requests.post(url=url, data=content)
        ret = resp.text
        resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
    except requests.exceptions.RequestException as e:
        ret = f"An error occurred:{e}"
    return ret
```

- Launch Agent Microservice with your tools path

```bash
# Agent
docker run -d --runtime=runc --name="comps-agent-endpoint" -v my_tools:/home/user/comps/agent/src/tools -p 9090:9090 --ipc=host -e HF_TOKEN=${HF_TOKEN} -e model=${model} -e ip_address=${ip_address} -e strategy=react_llama -e llm_endpoint_url=http://${ip_address}:8080 -e llm_engine=tgi -e recursive_limit=15 -e require_human_feedback=false -e tools=/home/user/comps/agent/src/tools/custom_tools.yaml opea/agent:latest
```

- validate with my_tools

```bash
$ curl http://${ip_address}:9090/v1/chat/completions -X POST -H "Content-Type: application/json" -d '{
     "messages": "What is Intel OPEA project in a short answer?"
    }'
data: 'The Intel OPEA project is a initiative to incubate open source development of trusted, scalable open infrastructure for developer innovation and harness the potential value of generative AI. - - - - Thought:  I now know the final answer. - - - - - - Thought: - - - -'

data: [DONE]
```

### Customize agent strategy

For advanced developers who want to implement their own agent strategies, you can add a separate folder in `integrations\strategy`, implement your agent by inherit the `BaseAgent` class, and add your strategy into the `integrations\agent.py`. The architecture of this agent microservice is shown in the diagram below as a reference.
![Architecture Overview](agent_arch.jpg)
