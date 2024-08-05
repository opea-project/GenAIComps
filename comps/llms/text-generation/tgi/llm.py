# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from fastapi.responses import StreamingResponse
from huggingface_hub import AsyncInferenceClient
from langsmith import traceable

from comps import (
    GeneratedDoc,
    LLMParamsDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

from comps.cores.proto.api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)
from template import ChatRagTemplate
from typing import Union
from openai import OpenAI

llm_endpoint = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")
llm = AsyncInferenceClient(
        model=llm_endpoint,
        timeout=600,
    )


@register_microservice(
    name="opea_service@llm_tgi",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9000,
)
@traceable(run_type="llm")
@register_statistics(names=["opea_service@llm_tgi"])
async def llm_generate(input: Union[LLMParamsDoc, ChatCompletionRequest]):

    stream_gen_time = []
    start = time.time()

    if isinstance(input, LLMParamsDoc):
        # apply default template for rag documnents
        if input.documents:
            if input.chat_template:
                if "{context}" in input.chat_template:
                    prompt = input.chat_template.format(question=input.query,
                                context="\n".join(input.documents))
                else:
                    prompt = input.chat_template.format(question=input.query)
            else:
                # use default template
                prompt = ChatRagTemplate.generate_prompt(input.query, input.documents)
        else:
            prompt = input.query

        text_generation = await llm.text_generation(
            prompt=prompt,
            stream=input.streaming,
            max_new_tokens=input.max_new_tokens,
            repetition_penalty=input.repetition_penalty,
            temperature=input.temperature,
            top_k=input.top_k,
            top_p=input.top_p,
        )
        if input.streaming:
            async def stream_generator():
                chat_response = ""
                async for text in text_generation:
                    stream_gen_time.append(time.time() - start)
                    chat_response += text
                    chunk_repr = repr(text.encode("utf-8"))
                    print(f"[llm - chat_stream] chunk:{chunk_repr}")
                    yield f"data: {chunk_repr}\n\n"
                print(f"[llm - chat_stream] stream response: {chat_response}")
                statistics_dict["opea_service@llm_tgi"].append_latency(stream_gen_time[-1], stream_gen_time[0])
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            statistics_dict["opea_service@llm_tgi"].append_latency(time.time() - start, None)
            return GeneratedDoc(text=text_generation, prompt=input.query)

    else:
        client = OpenAI(
            api_key="EMPTY",
            base_url=llm_endpoint + "/v1",
        )

        if isinstance(input.messages, str):
            prompt = input.messages
            if input.chat_template is not None:
                if "{context}" in input.chat_template:
                    if input.documents is None or input.documents == []:
                        prompt = input.chat_template.format(question=input.messages, context="")
                    else:
                        prompt = input.chat_template.format(question=input.messages,
                                context="\n".join(input.documents))
                else:
                    prompt = input.chat_template.format(question=input.messages)

            chat_completion = client.completions.create(
                model="tgi",
                prompt=prompt,
                best_of=input.best_of,
                echo=input.echo,
                frequency_penalty=input.frequency_penalty,
                logit_bias=input.logit_bias,
                logprobs=input.logprobs,
                max_tokens=input.max_tokens,
                n=input.n,
                presence_penalty=input.presence_penalty,
                seed=input.seed,
                stop=input.stop,
                stream=input.stream,
                suffix=input.suffix,
                temperature=input.temperature,
                top_p=input.top_p,
                user=input.user,
                )
        else:
            if input.messages[0]["role"] == "system":
                if "{context}" in input.messages[0]["content"]:
                    if input.documents is None or input.documents == []:
                        input.messages[0]["content"].format(context="")
                    else:
                        input.messages[0]["content"].format(context="\n".join(input.documents))
            else:
                if input.chat_template is not None:
                    system_prompt = input.chat_template
                    if "{context}" in system_prompt:
                        if input.documents is None or input.documents == []:
                            system_prompt = system_prompt.format(context="")
                        else:
                            system_prompt = system_prompt.format(context="\n".join(input.documents))

                    input.messages.insert(0,
                        {"role": "system",
                        "content": system_prompt}
                    )

            chat_completion = client.chat.completions.create(
                model="tgi",
                messages=input.messages,
                frequency_penalty=input.frequency_penalty,
                logit_bias=input.logit_bias,
                logprobs=input.logprobs,
                top_logprobs=input.top_logprobs,
                max_tokens=input.max_tokens,
                n=input.n,
                presence_penalty=input.presence_penalty,
                response_format=input.response_format,
                seed=input.seed,
                service_tier=input.service_tier,
                stop=input.stop,
                stream=input.stream,
                stream_options=input.stream_options,
                temperature=input.temperature,
                top_p=input.top_p,
                tools=input.tools,
                tool_choice=input.tool_choice,
                parallel_tool_calls=input.parallel_tool_calls,
                user=input.user,
            )

        if input.stream:
            def stream_generator():
                for c in chat_completion:
                    print(c)
                    yield f"data: {c.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            return chat_completion

if __name__ == "__main__":
    opea_microservices["opea_service@llm_tgi"].start()
