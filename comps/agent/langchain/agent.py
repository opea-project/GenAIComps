# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pathlib
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from tqdm import tqdm

cur_path = pathlib.Path(__file__).parent.resolve()
comps_path = os.path.join(cur_path, "../../../")
sys.path.append(comps_path)

from comps import GeneratedDoc, LLMParamsDoc, ServiceType, opea_microservices, register_microservice
from comps.agent.langchain.src.agent import instantiate_agent
from comps.agent.langchain.src.utils import format_date, get_args


@register_microservice(
    name="opea_service@comps-react-agent",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9090,
)
def llm_generate(input: LLMParamsDoc):
    # 1. initialize the agent
    args, _ = get_args()
    config = {"recursion_limit": args.recursion_limit}
    agent_inst = instantiate_agent(args)
    app = agent_inst.app

    # 2. prepare the input for the agent
    # assuming you have Ollama installed and have llama3 model pulled with `ollama pull llama3`
    if input.streaming:

        async def stream_generator():
            initial_state = {
                "input": input.query,
                "date": datetime.today().strftime("%Y-%m-%d"),
                "plan_errors": [],
                "past_steps": [],
            }
            async for chunk in app.astream(initial_state, config=config):
                # Agent Action
                if "actions" in chunk:
                    for action in chunk["actions"]:
                        print(f"Calling Tool: `{action.tool}` with input `{action.tool_input}`")
                # Observation
                elif "steps" in chunk:
                    for step in chunk["steps"]:
                        print(f"Tool Result: `{step.observation}`")
                # Final result
                elif "output" in chunk:
                    yield f"data: {repr(chunk['output'])}\n\n"
                else:
                    raise ValueError()
                print("---")
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:

        async def stream_generator():
            initial_state = {
                "input": input.query,
                "date": datetime.today().strftime("%Y-%m-%d"),
                "plan_errors": [],
                "past_steps": [],
            }
            for event in app.stream(initial_state, config=config):
                for k, v in event.items():
                    print("{}: {}".format(k, v))
                yield f"data: {repr(event)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    opea_microservices["opea_service@comps-react-agent"].start()
