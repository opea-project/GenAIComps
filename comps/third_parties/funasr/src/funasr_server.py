# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import os
import uuid
from typing import List

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import Response
from funasr_paraformer_model import Paraformer
from pydub import AudioSegment
from starlette.middleware.cors import CORSMiddleware

from comps import CustomLogger
from comps.cores.proto.api_protocol import AudioTranscriptionResponse

logger = CustomLogger("funasr_paraformer")
logflag = os.getenv("LOGFLAG", False)

app = FastAPI()
asr = None

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/v1/asr")
async def audio_to_text(request: Request):
    logger.info("Paraformer generation begin.")
    uid = str(uuid.uuid4())
    file_name = uid + ".wav"
    request_dict = await request.json()
    audio_b64_str = request_dict.pop("audio")
    with open(file_name, "wb") as f:
        f.write(base64.b64decode(audio_b64_str))

    try:
        asr_result = asr.transcribe(file_name)
    except Exception as e:
        logger.error(e)
        asr_result = e
    finally:
        os.remove(file_name)
    return {"asr_result": asr_result}


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),  # Handling the uploaded file directly
    model: str = Form("paraformer-zh"),
    language: str = Form("english"),
    prompt: str = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0),
    timestamp_granularities: List[str] = Form(None),
):
    logger.info("Paraformer generation begin.")
    audio_content = await file.read()
    # validate the request parameters
    if model != asr.model_name:
        raise Exception(f"ASR model mismatch! Please make sure you pass --model_name to {model}")
    if prompt is not None or response_format != "json" or temperature != 0 or timestamp_granularities is not None:
        logger.warning(
            "Currently parameters 'prompt', 'response_format', 'temperature', 'timestamp_granularities' are not supported!"
        )

    uid = str(uuid.uuid4())
    file_name = uid + ".wav"
    # Save the uploaded file
    with open(file_name, "wb") as buffer:
        buffer.write(audio_content)

    try:
        asr_result = asr.transcribe(file_name)
    except Exception as e:
        logger.error(e)
        asr_result = e
    finally:
        os.remove(file_name)

    return AudioTranscriptionResponse(text=asr_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7066)
    parser.add_argument("--model_name_or_path", type=str, default="paraformer-zh")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--revision", type=str, default="v2.0.4")

    args = parser.parse_args()
    asr = Paraformer(
        model_name=args.model_name_or_path,
        device=args.device,
        revision=args.revision,
    )

    uvicorn.run(app, host=args.host, port=args.port)
