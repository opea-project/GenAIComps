# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identified: Apache-2.0


import argparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import uvicorn

from predictionguard import PredictionGuard


client = PredictionGuard()

app = FastAPI()


@app.post("/generate")
async def generate(request: Request) -> Response:
    request_dict = await request.json()

    prompt = request_dict.pop("prompt")
    img_b64_str = request_dict.pop("img_b64_str")
    max_new_tokens = request_dict.pop("max_new_tokens", 100)

    # make a request to the Prediction Guard API using the LlaVa model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": img_b64_str}},
            ],
        },
    ]
    result = client.chat.completions.create(
        model="llava-1.5-7b-hf",
        messages=messages,
        max_tokens=max_new_tokens,
    )

    response = {"text": result["choices"][0]["message"]["content"]}

    return JSONResponse(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8399)

    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
    )
