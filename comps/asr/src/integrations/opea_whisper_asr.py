# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import aiohttp
from typing import List
import os
import requests
from fastapi import File, Form, UploadFile

from comps import OpeaComponent, CustomLogger, ServiceType
from comps.cores.proto.api_protocol import (
    AudioTranscriptionResponse,
)

logger = CustomLogger("opea_whisper_asr")
logflag = os.getenv("LOGFLAG", False)


class OpeaWhisperAsr(OpeaComponent):
    """
    A specialized ASR (Automatic Speech Recognition) component derived from OpeaComponent for Whisper ASR services.

    Attributes:
        model_name (str): The name of the ASR model used.
    """
    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.ASR.name.lower(), description, config)
        self.base_url = os.getenv("ASR_ENDPOINT", "http://localhost:7066/v1/audio/transcriptions")

    async def invoke(
            self,
            file: UploadFile = File(...),  # Handling the uploaded file directly
            model: str = Form("openai/whisper-small"),
            language: str = Form("english"),
            prompt: str = Form(None),
            response_format: str = Form("json"),
            temperature: float = Form(0),
            timestamp_granularities: List[str] = Form(None),
        ) -> AudioTranscriptionResponse:
        """
        Invole the ASR service to generate transcription for the provided input.
        """ 
        # Read the uploaded file
        file_contents = await file.read()

        # Prepare the files and data for requests.post
        files = {
            "file": (file.filename, file_contents, file.content_type),
        }
        data = {
            "model": model,
            "language": language,
            "prompt": prompt,
            "response_format": response_format,
            "temperature": temperature,
            "timestamp_granularities": timestamp_granularities,
        }

        # Send the file and model to the server
        response = requests.post(self.base_url, files=files, data=data)
        res = response.json()["text"]
        return AudioTranscriptionResponse(text=res)


    async def check_health(self) -> bool:
        """
        Checks the health of the embedding service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            async with aiohttp.ClientSession() as client:
                async with client.get(f"{self.base_url}/health") as response:
                    # If status is 200, the service is considered alive
                    if response.status == 200:
                        return True
        except aiohttp.ClientError as e:
            # Handle connection errors, timeouts, etc.
            print(f"Health check failed: {e}")
        return False