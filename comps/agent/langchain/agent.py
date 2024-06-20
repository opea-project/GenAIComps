# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pathlib
import sys
from pathlib import Path

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from typing import List
from tqdm import tqdm
from datetime import datetime

cur_path = pathlib.Path(__file__).parent.resolve()
comps_path = os.path.join(cur_path, "../../../")
sys.path.append(comps_path)

from comps.agent.langchain.src.utils import get_args, format_date
from comps import GeneratedDoc, LLMParamsDoc, ServiceType, opea_microservices, register_microservice
from comps.agent.langchain.src.agent import instantiate_agent

@register_microservice(
    name="opea_service@comps-react-agent", service_type=ServiceType.LLM, endpoint="/v1/chat/completions", host="0.0.0.0", port=9000
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
                "date":datetime.today().strftime('%Y-%m-%d'),
                "plan_errors":[],
                "past_steps":[]
            }
            for event in app.stream(initial_state, config=config):
                for k, v in event.items():
                    print("{}: {}".format(k,v))
                yield f"data: {repr(event)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        response = agent_inst.invoke(input.query)
        return GeneratedDoc(text=response, prompt=input.query)


if __name__ == "__main__":
    opea_microservices["opea_service@comps-react-agent"].start()