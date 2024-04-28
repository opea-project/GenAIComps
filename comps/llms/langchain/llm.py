# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

from docarray.base_doc import DocArrayResponse
from http_service import HTTPService
from langchain_community.llms import HuggingFaceEndpoint

from comps import GeneratedDoc, LLMParamsDoc, TextDoc


async def setup():
    runtime_args = {
        "title": "LLM microservice",
        "description": "This is an example of LLM microservice.",
        "protocol": "http",
        "port": 8090,
        "host": "localhost",
    }
    service = HTTPService(runtime_args=runtime_args, cors=False)
    app = service.app

    @app.post(
        path="/v1/generate",
        response_model=GeneratedDoc,
        response_class=DocArrayResponse,
        summary="Get the generated text of LLM inference.",
        tags=["Debug"],
    )
    def llm_generate(input: TextDoc, entrypoint: TextDoc = None, params: LLMParamsDoc = None) -> GeneratedDoc:
        entrypoint = entrypoint.text if entrypoint else "http://localhost:8080"
        params = params if params else LLMParamsDoc()
        llm = HuggingFaceEndpoint(
            endpoint_url=entrypoint,
            max_new_tokens=params.max_new_tokens,
            top_k=params.top_k,
            top_p=params.top_p,
            typical_p=params.typical_p,
            temperature=params.temperature,
            repetition_penalty=params.repetition_penalty,
            streaming=params.streaming,
        )
        response = llm(input.text)
        res = GeneratedDoc(text=response, prompt=input.text)
        return res

    await service.initialize_server()
    await service.execute_server()


asyncio.run(setup())
