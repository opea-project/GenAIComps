# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import uuid
import wave

from fastapi import WebSocket
from faster_whisper import WhisperModel
from whisper_model import get_nlp

BYTES_PER_SAMPLE = 2  # 16-bit audio
SAMPLE_RATE = 16000  # Hz


class FasterWhisperModel:
    """Leverage FasterWhisper to do realtime transcription on CPU"""

    def __init__(
        self,
        model_size_or_path="tiny",
        device="cpu",
        compute_type="int8"
    ) -> None:
        self.model = WhisperModel(
            model_size_or_path=model_size_or_path,
            device=device,
            compute_type=compute_type,
            local_files_only=False
        )
        self.model_size_or_path = model_size_or_path

    async def audio2text_streaming(self,
                                   websocket: WebSocket,
                                   audio_data: bytes,
                                   timestamp: list[dict],
                                   item_id: str,
                                   language: str = "en"):
        """process the audio data chunk"""
        if audio_data is None or len(audio_data) == 0:
            await websocket.send_json({
                "event_id": "event_0",
                "type": "error",
                "error": {
                    "type": "empty_audio_data_error",
                    "code": "empty_audio_data_error",
                    "message": "Received empty data for streaming asr.",
                    "param": None,
                    "event_id": "event_0"
                }
            })
            return

        # get nlp for sentence segmentation
        nlp = get_nlp(language)

        # generate a temporary file name for the segment
        uid = f"{uuid.uuid4().hex[:12]}"
        temp_filename = f"{uid}.wav"

        def process_transcription(filename: str, language: str):
            return self.model.transcribe(
                filename,
                language=None if language == "auto" else language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500)
            )

        try:
            # create a WAV file
            with wave.open(temp_filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # single channel
                wav_file.setsampwidth(BYTES_PER_SAMPLE)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_data)

            print("started transcription...")
            # use faster-whisper for speech recognition
            segments, _ = await asyncio.to_thread(process_transcription, temp_filename, language)

            # get all segments and use stanza to further segment to words and sentences
            # to match openai format
            segments = list(segments)
            for segment in segments:
                print("[%.2fs -> %.2fs] %s" %
                      (segment.start, segment.end, segment.text))
                for sent in nlp(segment.text).sentences:
                    sent_id = f"event_{uuid.uuid4().hex[:12]}"
                    for token in sent.words:
                        id = f"event_{uuid.uuid4().hex[:12]}"
                        delta_resp = {
                            "event_id": id,
                            "type": "conversation.item.input_audio_transcription.delta",
                            "item_id": item_id,
                            "context_index": 0,
                            "delta": token.text
                        }
                        await websocket.send_json(delta_resp)

                    complete_resp = {
                        "event_id": sent_id,
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": item_id,
                        "context_index": 0,
                        "transcript": sent.text
                    }
                    await websocket.send_json(complete_resp)

        except Exception as e:
            print(f"Error in transcription: {e}")
            await websocket.send_json({
                "status": "error",
                "error": str(e)
            })
        finally:
            # clean up the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
