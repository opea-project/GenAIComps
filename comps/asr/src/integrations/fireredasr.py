#!/usr/bin/env python3

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import tempfile
import time
import uuid
from typing import List, Union
from pathlib import Path

import requests
from fastapi import File, Form, UploadFile

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import AudioTranscriptionResponse

logger = CustomLogger("opea_fireredasr")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("OPEA_FIREREDASR_ASR")
class OpeaFireRedAsr(OpeaComponent):
    """
    A specialized ASR (Automatic Speech Recognition) component for FireRedASR services.
    
    This component integrates FireRedASR model with OPEA microservice architecture.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.ASR.name.lower(), description, config)
        self.model_dir = os.getenv("FIREREDASR_MODEL_DIR", "/app/pretrained_models")
        self.asr_type = os.getenv("FIREREDASR_ASR_TYPE", "llm")  # "aed" or "llm"
        
        # Initialize FireRedASR model
        self.model = None
        self._initialize_model()
        
        # Health check
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaFireRedAsr health check failed.")

    def _initialize_model(self):
        """Initialize FireRedASR model."""
        try:
            # Add FireRedASR to Python path
            import sys
            fireredasr_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "FireRedASR", "fireredasr")
            if fireredasr_path not in sys.path:
                sys.path.insert(0, fireredasr_path)
            
            # Import FireRedASR
            from fireredasr.models.fireredasr import FireRedAsr
            
            # Load model
            self.model = FireRedAsr.from_pretrained(self.asr_type, self.model_dir)
            logger.info(f"FireRedASR model loaded successfully: {self.asr_type} from {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FireRedASR model: {e}")
            raise

    async def invoke(
        self,
        file: Union[str, UploadFile],
        model: str = "fireredasr",
        language: str = "auto",
        prompt: str = None,
        response_format: str = "json",
        temperature: float = 0.0,
        timestamp_granularities: List[str] = None,
        fireredasr_model_dir: str = None,
        fireredasr_asr_type: str = None,
    ) -> AudioTranscriptionResponse:
        """
        Invoke FireRedASR service to generate transcription for the provided audio file.
        
        Args:
            file: Audio file (UploadFile or base64 string)
            model: Model name (currently only "fireredasr" is supported)
            language: Language code (currently "auto" is supported)
            prompt: Optional prompt for transcription
            response_format: Response format ("json" or "text")
            temperature: Temperature parameter (currently not used in FireRedASR)
            timestamp_granularities: Timestamp granularities (currently not supported)
            fireredasr_model_dir: Model directory path
            fireredasr_asr_type: ASR type ("aed" or "llm")
            
        Returns:
            AudioTranscriptionResponse: Transcription result
        """
        if fireredasr_model_dir:
            self.model_dir = fireredasr_model_dir
        if fireredasr_asr_type:
            self.asr_type = fireredasr_asr_type
            
        # Re-initialize model if parameters changed
        if self.model is None or self.model_dir != fireredasr_model_dir or self.asr_type != fireredasr_asr_type:
            self._initialize_model()

        if isinstance(file, str):
            # Handle base64 encoded audio
            return await self._handle_base64_audio(file, language, prompt)
        else:
            # Handle uploaded file
            return await self._handle_uploaded_file(file, language, prompt)

    async def _handle_base64_audio(self, audio_base64: str, language: str, prompt: str) -> AudioTranscriptionResponse:
        """Handle base64 encoded audio."""
        try:
            # Save base64 audio to temporary file
            import base64
            
            # Decode base64
            audio_data = base64.b64decode(audio_base64)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Process audio file
                result = await self._process_audio_file(temp_file_path, language, prompt)
                return AudioTranscriptionResponse(text=result["text"])
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error processing base64 audio: {e}")
            raise

    async def _handle_uploaded_file(self, file: UploadFile, language: str, prompt: str) -> AudioTranscriptionResponse:
        """Handle uploaded audio file."""
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(suffix=f".{file.filename.split('.')[-1] if '.' in file.filename else '.wav'}", delete=False) as temp_file:
                file_content = await file.read()
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                # Process audio file
                result = await self._process_audio_file(temp_file_path, language, prompt)
                return AudioTranscriptionResponse(text=result["text"])
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            raise

    async def _process_audio_file(self, audio_path: str, language: str, prompt: str) -> dict:
        """Process audio file using FireRedASR model."""
        try:
            # Prepare model arguments based on ASR type
            model_args = {
                "use_gpu": os.getenv("FIREREDASR_USE_GPU", "1").lower() == "true",
                "batch_size": int(os.getenv("FIREREDASR_BATCH_SIZE", "1")),
                "beam_size": int(os.getenv("FIREREDASR_BEAM_SIZE", "1")),
            }
            
            # Add type-specific arguments
            if self.asr_type == "aed":
                model_args.update({
                    "nbest": int(os.getenv("FIREREDASR_NBEST", "1")),
                    "softmax_smoothing": float(os.getenv("FIREREDASR_SOFTMAX_SMOOTHING", "1.0")),
                    "aed_length_penalty": float(os.getenv("FIREREDASR_AED_LENGTH_PENALTY", "0.0")),
                    "eos_penalty": float(os.getenv("FIREREDASR_EOS_PENALTY", "1.0")),
                    "decode_max_len": int(os.getenv("FIREREDASR_DECODE_MAX_LEN", "0")),
                })
            elif self.asr_type == "llm":
                model_args.update({
                    "decode_min_len": int(os.getenv("FIREREDASR_DECODE_MIN_LEN", "0")),
                    "repetition_penalty": float(os.getenv("FIREREDASR_REPETITION_PENALTY", "1.0")),
                    "llm_length_penalty": float(os.getenv("FIREREDASR_LLM_LENGTH_PENALTY", "0.0")),
                    "temperature": float(os.getenv("FIREREDASR_TEMPERATURE", "1.0")),
                })
            
            # Generate unique ID for the audio file
            uttid = str(uuid.uuid4())
            
            # Transcribe audio
            start_time = time.time()
            results = self.model.transcribe([uttid], [audio_path], model_args)
            elapsed_time = time.time() - start_time
            
            if logflag:
                logger.info(f"FireRedASR transcription completed in {elapsed_time:.2f}s")
            
            if results and len(results) > 0:
                return {
                    "text": results[0]["text"],
                    "uttid": results[0]["uttid"],
                    "rtf": results[0].get("rtf", "N/A"),
                    "processing_time": elapsed_time
                }
            else:
                raise Exception("No transcription results returned")
                
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            raise

    def check_health(self) -> bool:
        """
        Check the health of the FireRedASR service.
        
        Returns:
            bool: True if the service is healthy, False otherwise
        """
        try:
            # Check if model directory exists
            if not os.path.exists(self.model_dir):
                logger.error(f"Model directory does not exist: {self.model_dir}")
                return False
            
            # Check if model files exist
            required_files = ["model.pth.tar", "cmvn.ark"]
            if self.asr_type == "aed":
                required_files.extend(["dict.txt", "train_bpe1000.model"])
            elif self.asr_type == "llm":
                required_files.extend(["asr_encoder.pth.tar", "Qwen2-7B-Instruct"])
            
            for file in required_files:
                file_path = os.path.join(self.model_dir, file)
                if not os.path.exists(file_path):
                    logger.error(f"Required model file not found: {file_path}")
                    return False
            
            # Try to load a small test
            try:
                # This is a basic health check - we'll try to initialize the model
                # If it works, we're healthy
                test_model = self.model
                if test_model is not None:
                    return True
            except Exception as e:
                logger.error(f"Model health check failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        return False