# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import base64
import io
import os
import sys
import threading
import time
import uuid
from typing import List, Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import Body, FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from faster_whisper_model import FasterWhisperModel
from pydub import AudioSegment
from starlette.middleware.cors import CORSMiddleware
from whisper_model import WhisperModelCPU, WhisperModelHPU, WhisperModelXPU, get_nlp

from comps import CustomLogger
from comps.cores.proto.api_protocol import AudioTranscriptionResponse

logger = CustomLogger("whisper")
logflag = os.getenv("LOGFLAG", False)

# websocket configuration constants
DEFAULT_SESSION_TIMEOUT_MINS = 30  # set 30 mins for session timeout

app = FastAPI()
asr = None
streaming_asr = None
streaming_asr_ready = threading.Event()
streaming_asr_model = ""
streaming_asr_device = ""

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


def init_streaming_whisper(
    streaming_model_size: str, model_size: str, device: str, return_timestamps: str, language: str
):
    """Init streaming whisper service in a thread."""
    global streaming_asr
    if device == "cpu":
        # for cpu device, leverage the faster-whisper library for streaming asr
        # notice that the library uses different naming for the model definition
        try:
            prefix = "openai/whisper-"
            formatted_model = args.streaming_model_name_or_path.replace(prefix, "", 1)
            streaming_asr = FasterWhisperModel(
                model_size_or_path=formatted_model,
            )
            streaming_asr_ready.set()
            logger.info("Faster Whisper model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Faster Whisper model: {e}")
            streaming_asr_ready.set()  # set the event even if it fails to avoid permanent waiting
    # for other devices, check if the streaming asr uses the same model as the normal one
    # if yes, skip model initialization
    elif model_size == streaming_model_size:
        streaming_asr = asr
        streaming_asr_ready.set()
    else:
        if device == "hpu":
            try:
                streaming_asr = WhisperModelHPU(
                    model_name_or_path=streaming_model_size,
                    language=language,
                    device=device,
                    return_timestamps=return_timestamps,
                )
                streaming_asr_ready.set()
            except Exception as e:
                logger.error(f"Failed to initialize streaming Whisper model on hpu: {e}")
                streaming_asr_ready.set()

        elif device == "xpu":
            try:
                streaming_asr = WhisperModelXPU(
                    model_name_or_path=streaming_model_size,
                    language=language,
                    device=device,
                    return_timestamps=return_timestamps,
                )
                streaming_asr_ready.set()
            except Exception as e:
                logger.error(f"Failed to initialize streaming Whisper model on xpu: {e}")
                streaming_asr_ready.set()
    # load the segmentation tool
    get_nlp(language)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/v1/asr")
async def audio_to_text(request: Request):
    logger.info("Whisper generation begin.")
    uid = str(uuid.uuid4())
    file_name = uid + ".wav"
    request_dict = await request.json()
    audio_b64_str = request_dict.pop("audio")
    with open(file_name, "wb") as f:
        f.write(base64.b64decode(audio_b64_str))

    audio = AudioSegment.from_file(file_name)
    audio = audio.set_frame_rate(16000)

    audio.export(f"{file_name}", format="wav")
    try:
        asr_result = asr.audio2text(file_name)
    except Exception as e:
        logger.error(e)
        asr_result = e
    finally:
        os.remove(file_name)
    return {"asr_result": asr_result}


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),  # Handling the uploaded file directly
    model: str = Form("openai/whisper-small"),
    language: str = Form("english"),
    prompt: str = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0),
    timestamp_granularities: List[str] = Form(None),
):
    logger.info("Whisper generation begin.")
    audio_content = await file.read()
    # validate the request parameters
    if model != asr.asr_model_name_or_path:
        raise Exception(
            f"ASR model mismatch! Please make sure you pass --model_name_or_path or set environment variable ASR_MODEL_PATH to {model}"
        )
    asr.language = language
    if prompt is not None or response_format != "json" or temperature != 0 or timestamp_granularities is not None:
        logger.warning(
            "Currently parameters 'language', 'response_format', 'temperature', 'timestamp_granularities' are not supported!"
        )

    uid = str(uuid.uuid4())
    file_name = uid + ".wav"
    # Save the uploaded file
    with open(file_name, "wb") as buffer:
        buffer.write(audio_content)

    audio = AudioSegment.from_file(file_name)
    audio = audio.set_frame_rate(16000)

    audio.export(f"{file_name}", format="wav")

    try:
        asr_result = asr.audio2text(file_name)
    except Exception as e:
        logger.error(e)
        asr_result = e
    finally:
        os.remove(file_name)

    return AudioTranscriptionResponse(text=asr_result)


@app.post("/v1/realtime/transcription_sessions")
async def create_realtime_transcription_session(config: Optional[dict] = Body(default=None)):
    # set default values for all configs
    client_secret = None
    turn_detection = {"silence_duration_ms": 500, "prefix_padding_ms": 300, "threshold": 0.5, "type": "server_vad"}
    input_audio_format = "pcm16"
    input_audio_transcription = {"model": "openai/whisper-medium"}
    modalities = ["audio", "text"]

    if config:
        if config.get("input_audio_format"):
            input_audio_format = config["input_audio_format"]
        if config.get("turn_detection"):
            td = config["turn_detection"]
            turn_detection = {
                "silence_duration_ms": td.get("silence_duration_ms", 500),
                "prefix_padding_ms": td.get("prefix_padding_ms", 300),
                "threshold": td.get("threshold", 0.5),
                "type": td.get("type", "server_vad"),
            }
        if config.get("input_audio_transcription"):
            input_audio_transcription = config["input_audio_transcription"]
        if config.get("modalities"):
            modalities = config["modalities"]
        if config.get("include"):
            logger.info("Unused Include param for now.")

    if input_audio_transcription.get("model") != streaming_asr_model:
        logger.info("Unmatched model for streaming asr. Fallback to default model")
        input_audio_transcription["model"] = streaming_asr_model

    """
    # user may choose from the available models and start the streaming whisper thread later
    available_streaming_models = ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small", "openai/whisper-medium"]
    if input_audio_transcription.model in available_streaming_models:
        streaming_whisper_thread = threading.Thread(
            target=init_streaming_whisper,
            args=(input_audio_transcription.model, args.model_name_or_path,
                args.device, args.return_timestamps, args.language),
            daemon=True
        )
        streaming_whisper_thread.start()
    else:
        logger.error("Unsupported model chosen, fallback to default streaming model")
    """
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    expires_at = int(time.time()) + 1800  # expires in 30 mins

    session = {
        "id": session_id,
        "object": "realtime.transcription_session",
        "expires_at": expires_at,
        "client_secret": client_secret,
        "input_audio_format": input_audio_format,
        "input_audio_transcription": input_audio_transcription,
        "turn_detection": turn_detection,
        "modalities": modalities,
    }

    return session


@app.websocket("/v1/realtime")
async def audio_transcriptions_streaming(websocket: WebSocket, intent: str = "transcription"):
    """This endpoint is used to stream the transcription of the audio input.

    Args:
        websocket: WebSocket
        intent: String, default is "transcription"
    """
    if intent != "transcription":
        logger.warning("Unsupported function. Currently only support the 'transcription' intent.")
        await websocket.close()
        return

    await websocket.accept()

    # wait for the streaming model to be initialized
    if not streaming_asr_ready.wait(timeout=10):
        await websocket.send_json(
            {
                "event_id": "event_0",
                "type": "error",
                "error": {
                    "type": "initialization_timeout",
                    "code": "initialization_timeout",
                    "message": "The streaming asr failed to initialize.",
                    "param": None,
                    "event_id": "event_0",
                },
            }
        )
        return

    if streaming_asr is None:
        await websocket.send_json(
            {
                "event_id": "event_0",
                "type": "error",
                "error": {
                    "type": "initialization_failure",
                    "code": "initialization_failure",
                    "message": "The streaming asr model failed to initialize.",
                    "param": None,
                    "event_id": "event_0",
                },
            }
        )
        return

    try:
        # initialize audio buffer
        event_id = f"event_{uuid.uuid4().hex[:12]}"
        # initialize VAD model for non CPU cases
        get_speech_timestamps = None
        if streaming_asr_device != "cpu":
            import torch

            torch.set_num_threads(1)
            model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
            get_speech_timestamps, _, _, _, _ = utils

        while True:
            try:
                # receive message with timeout
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_json(), timeout=DEFAULT_SESSION_TIMEOUT_MINS * 60
                    )
                except asyncio.TimeoutError:
                    logger.info("Session expired after 30 minutes.")
                    await websocket.send_json(
                        {
                            "event_id": event_id,
                            "type": "error",
                            "error": {
                                "type": "expired_session_error",
                                "code": "expired_session_error",
                                "message": "session_expired - Your session hit the maximum duration of 30 minutes.",
                                "param": None,
                                "event_id": event_id,
                            },
                        }
                    )
                    await websocket.close()
                    break

                # process the message if it is a input_audio_buffer.append operand
                # shall support other functions later
                if message.get("type") == "input_audio_buffer.append":
                    item_id = f"item_{uuid.uuid4().hex[:12]}"
                    event_id = f"event_{uuid.uuid4().hex[:12]}"
                    speech_timestamps = None

                    # fetch audio data and pre-process the audio data
                    audio = base64.b64decode(message.get("audio", ""))

                    # process VAD in non-CPU cases
                    # for cpu, faster-whisper lib will handle the VAD support
                    if streaming_asr_device != "cpu":
                        audio_buffer = io.BytesIO(audio)
                        audio, sr = sf.read(audio_buffer)
                        if len(audio.shape) > 1:
                            audio = np.mean(audio, axis=1)
                        if sr != 16000:
                            import librosa

                            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                            sr = 16000
                        audio_int16 = (audio * 32767).astype(np.int16)
                        audio_tensor = torch.from_numpy(audio_int16)
                        audio_tensor = audio_tensor.squeeze()
                        speech_timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=16000)

                    # process transcription
                    await streaming_asr.audio2text_streaming(
                        websocket=websocket, audio_data=audio, timestamp=speech_timestamps, item_id=item_id
                    )
                else:
                    logger.info("Unsupported message type： %s", message.get("type"))
                    await websocket.send_json(
                        {
                            "event_id": "event_0",
                            "type": "error",
                            "error": {
                                "type": "unsupported_action_type",
                                "code": "unsupported_action_type",
                                "message": "Received unsupported action item.",
                                "param": None,
                                "event_id": "event_0",
                            },
                        }
                    )

            except WebSocketDisconnect:
                logger.info("WebSocket connection closed unexpectedly")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket connection: {e}")
                await websocket.send_json({"status": "error", "error": str(e)})
                break

    except Exception as e:
        logger.error(f"Error in audio streaming: {e}")
        await websocket.send_json({"status": "error", "error": str(e)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7066)
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-small")
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--return_timestamps", type=str, default=True)
    parser.add_argument("--enable_streaming", type=bool, default=False)
    parser.add_argument("--streaming_model_name_or_path", type=str, default="openai/whisper-small")

    args = parser.parse_args()
    # initiate asr according to device type
    if args.device == "cpu":
        asr = WhisperModelCPU(
            model_name_or_path=args.model_name_or_path,
            language=args.language,
            device=args.device,
            return_timestamps=args.return_timestamps,
        )
    elif args.device == "hpu":
        asr = WhisperModelHPU(
            model_name_or_path=args.model_name_or_path,
            language=args.language,
            device=args.device,
            return_timestamps=args.return_timestamps,
        )
    elif args.device == "xpu":
        asr = WhisperModelXPU(
            model_name_or_path=args.model_name_or_path,
            language=args.language,
            device=args.device,
            return_timestamps=args.return_timestamps,
        )
    else:
        # exit the process as there's unsupported device
        logger.error("Unsupported device type.")
        sys.exit(1)

    # check if streaming asr is enabled
    # if yes, initialize the asr_streaming instance in a separate thread
    if args.enable_streaming:
        streaming_asr_device = args.device
        streaming_asr_model = args.streaming_model_name_or_path
        streaming_whisper_thread = threading.Thread(
            target=init_streaming_whisper,
            args=(
                args.streaming_model_name_or_path,
                args.model_name_or_path,
                args.device,
                args.return_timestamps,
                args.language,
            ),
            daemon=True,
        )
        streaming_whisper_thread.start()
    uvicorn.run(app, host=args.host, port=args.port)
