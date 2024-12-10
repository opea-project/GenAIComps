# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identified: Apache-2.0
import json
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from predictionguard import PredictionGuard


from comps import (
    GeneratedDoc,
    LLMParamsDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

client = PredictionGuard()
app = FastAPI()


@register_microservice(
    name="opea_service@llm_predictionguard_docsum",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/docsum",
    host="0.0.0.0",
    port=9000,
)
@register_statistics(names=["opea_service@llm_predictionguard_docsum"])
def llm_generate(input: LLMParamsDoc):
    start = time.time()

    messages = [
        {
            "role": "system",
            "content": "You are a summarization assistant. Your goal is to provide a very concise, summarized responses of the user query.",
        },
        {"role": "user", "content": input.query},
    ]

    if input.streaming:

        async def stream_generator():
            chat_response = ""
            for res in client.chat.completions.create(
                model=input.model,
                messages=messages,
                max_tokens=input.max_tokens,
                temperature=input.temperature,
                top_p=input.top_p,
                top_k=input.top_k,
                stream=True,
            ):
                if "choices" in res["data"] and "delta" in res["data"]["choices"][0]:
                    delta_content = res["data"]["choices"][0]["delta"]["content"]
                    chat_response += delta_content
                    yield f"data: {delta_content}\n\n"
                else:
                    yield "data: [DONE]\n\n"

        statistics_dict["opea_service@llm_predictionguard_docsum"].append_latency(time.time() - start, None)
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        try:
            response = client.chat.completions.create(
                model=input.model,
                messages=messages,
                max_tokens=input.max_tokens,
                temperature=input.temperature,
                top_p=input.top_p,
                top_k=input.top_k,
            )

            print(json.dumps(
                response,
                sort_keys=True,
                indent=4,
                separators=(',', ': ')
            ))

            response_text = response["choices"][0]["message"]["content"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        statistics_dict["opea_service@llm_predictionguard_docsum"].append_latency(time.time() - start, None)
        return GeneratedDoc(text=response_text, prompt=input.query)


if __name__ == "__main__":
    opea_microservices["opea_service@llm_predictionguard_docsum"].start()
