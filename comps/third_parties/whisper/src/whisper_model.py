# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import threading
import time
import urllib.request
import uuid

import numpy as np
import torch
from datasets import Audio, Dataset
from fastapi import WebSocket
from pydub import AudioSegment
from transformers import WhisperForConditionalGeneration, WhisperProcessor

_nlp_cache = {}
_nlp_lock = threading.Lock()


def get_nlp(language: str = "en"):
    global _nlp_cache
    with _nlp_lock:
        if language not in _nlp_cache:
            import stanza

            stanza.download(language)
            _nlp_cache[language] = stanza.Pipeline(language)
        return _nlp_cache[language]


class BaseWhisperModel:
    """Base class for Whisper model."""

    processed_inputs = None
    waveform = None
    start_time = None
    processor = None

    def __init__(
        self,
        model_name_or_path="openai/whisper-small",
        language="english",
        device="cpu",
        hpu_max_len=8192,
        return_timestamps=False,
    ):
        """
        Args:
            model_name_or_path: the path to the model, e.g. openai/whisper-small
            language: the language of the model, e.g. english
            device: the device to use, e.g. cpu, xpu, hpu
            hpu_max_len: the maximum length of the input audio
        """
        self.device = device
        self.asr_model_name_or_path = os.environ.get("ASR_MODEL_PATH", model_name_or_path)
        print("Using model: {}".format(self.asr_model_name_or_path))

    def _audiosegment_to_librosawav(self, audiosegment):
        # https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentget_array_of_samples
        # This way is faster than librosa.load or HuggingFace Dataset wrapper
        # only select the first channel
        channel_sounds = audiosegment.split_to_mono()[:1]
        samples = [s.get_array_of_samples() for s in channel_sounds]

        fp_arr = np.array(samples).T.astype(np.float32)
        fp_arr /= np.iinfo(samples[0].typecode).max
        fp_arr = fp_arr.reshape(-1)

        return fp_arr

    def audio2text(self, audio_path):
        """Convert audio to text.
           Prepare the audio and processed_inputs for the model.

        Args:
            audio_path: the path to the input audio, e.g. ~/xxx.mp3
        """
        self.start_time = time.time()

        try:
            self.waveform = AudioSegment.from_file(audio_path).set_frame_rate(16000)
            self.waveform = self._audiosegment_to_librosawav(self.waveform)
        except Exception as e:
            print(f"[ASR] audiosegment to librosa wave fail: {e}")
            audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column("audio", Audio(sampling_rate=16000))
            self.waveform = audio_dataset[0]["audio"]["array"]

        try:
            self.processed_inputs = self.processor(
                self.waveform,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                return_attention_mask=True,
                sampling_rate=16000,
            )
        except RuntimeError as e:
            if "Padding size should be less than" in str(e):
                # short-form
                self.processed_inputs = self.processor(
                    self.waveform,
                    return_tensors="pt",
                    sampling_rate=16000,
                    return_attention_mask=True,
                )
            else:
                raise e
        if self.processed_inputs.input_features.shape[-1] < 3000:
            # short-form
            self.processed_inputs = self.processor(
                self.waveform,
                return_tensors="pt",
                sampling_rate=16000,
                return_attention_mask=True,
            )

    async def audio2text_streaming(
        self,
        websocket: WebSocket,
        audio_data: np.ndarray,
        timestamp: list[dict],
        item_id: str,
        language: str = "en",
        is_final: bool = False,
    ):
        """Convert audio to text in streaming mode.

        Args:
            audio_data: the input audio data in bytes
            websocket: the websocket used
            event_id: streaming event id
            item_id: item id for the streaming event
            language: language of the audio data
            is_final: boolean to show if this is the last piece of the streaming data
        """
        raise NotImplementedError("Subclasses must implement this method")


class WhisperModelCPU(BaseWhisperModel):
    """Whisper model for CPU."""

    def __init__(
        self,
        model_name_or_path="openai/whisper-small",
        language="english",
        device="cpu",
        hpu_max_len=8192,
        return_timestamps=False,
    ):
        super().__init__(model_name_or_path, language, device, hpu_max_len, return_timestamps)
        if self.device != "cpu":
            raise ValueError("WhisperModelCPU only supports CPU device")
        self.model = WhisperForConditionalGeneration.from_pretrained(self.asr_model_name_or_path).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(self.asr_model_name_or_path)
        self.model.eval()

        self.language = language
        self.hpu_max_len = hpu_max_len
        self.return_timestamps = return_timestamps

    def audio2text(self, audio_path):
        # get the audio and processed_inputs prepared
        super().audio2text(audio_path)
        predicted_ids = self.model.generate(
            **(
                self.processed_inputs.to(
                    self.device,
                )
            ),
            # language=self.language,
            return_timestamps=self.return_timestamps,
        )
        # pylint: disable=E1101
        result = self.processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)[0]
        if self.language in ["chinese", "mandarin"]:
            from zhconv import convert

            result = convert(result, "zh-cn")
        print(f"generated text in {time.time() - self.start_time} seconds, and the result is: {result}")
        return result

    async def audio2text_streaming(
        self,
        websocket: WebSocket,
        audio_data: np.ndarray,
        timestamp: list[dict],
        item_id: str,
        language: str = "en",
        is_final: bool = False,
    ):
        """Convert audio to text in streaming mode.

        Args:
            audio_data: the input audio data in bytes
            websocket: the websocket used
            event_id: streaming event id
            item_id: item id for the streaming event
            language: language of the audio data
            is_final: boolean to show if this is the last piece of the streaming data
        """
        raise NotImplementedError("Faster_whisper_model will handle the streaming part for CPU.")


class WhisperModelHPU(BaseWhisperModel):
    """Whisper model for HPU."""

    def __init__(
        self,
        model_name_or_path="openai/whisper-small",
        language="english",
        device="cpu",
        hpu_max_len=8192,
        return_timestamps=False,
    ):
        super().__init__(model_name_or_path, language, device, hpu_max_len, return_timestamps)
        if self.device != "hpu":
            raise ValueError("WhisperModelHPU only supports HPU device")

        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

        adapt_transformers_to_gaudi()
        self.model = WhisperForConditionalGeneration.from_pretrained(self.asr_model_name_or_path).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(self.asr_model_name_or_path)
        self.model.eval()

        self.language = language
        self.hpu_max_len = hpu_max_len
        self.return_timestamps = return_timestamps

        self._warmup_whisper_hpu_graph(os.path.dirname(os.path.abspath(__file__)) + "/assets/ljspeech_30s_audio.wav")
        self._warmup_whisper_hpu_graph(os.path.dirname(os.path.abspath(__file__)) + "/assets/ljspeech_60s_audio.wav")

    def _warmup_whisper_hpu_graph(self, path_to_audio):
        print("[ASR] warmup...")
        waveform = AudioSegment.from_file(path_to_audio).set_frame_rate(16000)
        waveform = self._audiosegment_to_librosawav(waveform)

        try:
            processed_inputs = self.processor(
                waveform,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                return_attention_mask=True,
                sampling_rate=16000,
            )
        except RuntimeError as e:
            if "Padding size should be less than" in str(e):
                # short-form
                processed_inputs = self.processor(
                    waveform,
                    return_tensors="pt",
                    sampling_rate=16000,
                    return_attention_mask=True,
                )
            else:
                raise e

        if processed_inputs.input_features.shape[-1] < 3000:
            # short-form
            processed_inputs = self.processor(
                waveform,
                return_tensors="pt",
                sampling_rate=16000,
                return_attention_mask=True,
            )
        else:
            processed_inputs["input_features"] = torch.nn.functional.pad(
                processed_inputs.input_features,
                (0, self.hpu_max_len - processed_inputs.input_features.size(-1)),
                value=-1.5,
            )
            processed_inputs["attention_mask"] = torch.nn.functional.pad(
                processed_inputs.attention_mask,
                (0, self.hpu_max_len + 1 - processed_inputs.attention_mask.size(-1)),
                value=0,
            )

        _ = self.model.generate(
            **(
                processed_inputs.to(
                    self.device,
                )
            ),
            language=self.language,
            return_timestamps=self.return_timestamps,
        )

    def audio2text(self, audio_path):
        super().audio2text(audio_path)
        if self.processed_inputs.input_features.shape[-1] > 3000:
            self.processed_inputs["input_features"] = torch.nn.functional.pad(
                self.processed_inputs.input_features,
                (0, self.hpu_max_len - self.processed_inputs.input_features.size(-1)),
                value=-1.5,
            )
            self.processed_inputs["attention_mask"] = torch.nn.functional.pad(
                self.processed_inputs.attention_mask,
                (0, self.hpu_max_len + 1 - self.processed_inputs.attention_mask.size(-1)),
                value=0,
            )

        predicted_ids = self.model.generate(
            **(
                self.processed_inputs.to(
                    self.device,
                )
            ),
            language=self.language,
            return_timestamps=self.return_timestamps,
        )
        # pylint: disable=E1101
        result = self.processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)[0]
        if self.language in ["chinese", "mandarin"]:
            from zhconv import convert

            result = convert(result, "zh-cn")
        print(f"generated text in {time.time() - self.start_time} seconds, and the result is: {result}")
        return result

    async def audio2text_streaming(
        self,
        websocket: WebSocket,
        audio_data: np.ndarray,
        timestamp: list[dict],
        item_id: str,
        language: str = "en",
        is_final: bool = False,
    ):
        """Convert audio to text in streaming mode.

        Args:
            audio_data: the input audio data in bytes
            websocket: the websocket used
            event_id: streaming event id
            item_id: item id for the streaming event
            language: language of the audio data
            is_final: boolean to show if this is the last piece of the streaming data
        """
        raise NotImplementedError("Not implemented yet.")


class WhisperModelXPU(BaseWhisperModel):
    """Whisper model for XPU."""

    def __init__(
        self,
        model_name_or_path="openai/whisper-small",
        language="english",
        device="xpu",
        hpu_max_len=8192,
        return_timestamps=False,
    ):
        super().__init__(model_name_or_path, language, device, hpu_max_len, return_timestamps)
        if self.device != "xpu":
            raise ValueError("WhisperModelXPU only supports XPU device")

        from ipex_llm.transformers import AutoModelForSpeechSeq2Seq

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name_or_path, load_in_4bit=True, optimize_model=False, use_cache=True
        )
        self.model.to("xpu")
        self.model.config.forced_decoder_ids = None
        self.processor = WhisperProcessor.from_pretrained(model_name_or_path)
        print("Whisper initialized on Intel XPU.")
        self.language = language
        self.return_timestamps = return_timestamps

    def audio2text(self, audio_path):
        """Convert audio to text.

        audio_path: the path to the input audio, e.g. ~/xxx.mp3
        """
        super().audio2text(audio_path)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.language, task="transcribe")
        with torch.inference_mode():
            predicted_ids = self.model.generate(
                **(
                    self.processed_inputs.to(
                        self.device,
                    )
                ),
                forced_decoder_ids=forced_decoder_ids,
            )
            # pylint: disable=E1101
            result = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        if self.language in ["chinese", "mandarin"]:
            from zhconv import convert

            result = convert(result, "zh-cn")
        print(f"generated text in {time.time() - self.start_time} seconds, and the result is: {result}")
        return result

    async def audio2text_streaming(
        self,
        websocket: WebSocket,
        audio_data: np.ndarray,
        timestamp: list[dict],
        item_id: str,
        language: str = "en",
        is_final: bool = False,
    ):
        """Convert audio to text in streaming mode.

        Args:
            audio_data: the input audio data in bytes
            websocket: the websocket used
            event_id: streaming event id
            item_id: item id for the streaming event
            language: language of the audio data
            is_final: boolean to show if this is the last piece of the streaming data
        """
        if audio_data is None or audio_data.size == 0:
            await websocket.send_json(
                {
                    "event_id": "event_0",
                    "type": "error",
                    "error": {
                        "type": "empty_audio_data_error",
                        "code": "empty_audio_data_error",
                        "message": "Received empty data for streaming asr.",
                        "param": None,
                        "event_id": "event_0",
                    },
                }
            )
            return
        nlp = get_nlp(language=language)

        def handle_transcription_data(audio):
            with torch.inference_mode():
                inputs = self.processor(
                    filtered_audio_data, sampling_rate=16000, return_attention_mask=True, return_tensors="pt"
                )
                input_features = inputs.input_features.to("xpu")
                attention_mask = inputs.attention_mask.to("xpu")
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
                predicted_ids = self.model.generate(
                    input_features, forced_decoder_ids=forced_decoder_ids, attention_mask=attention_mask
                )
                output_str = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                return output_str

        for ts in timestamp:
            filtered_audio_data = audio_data[int(ts["start"]) : int(ts["end"])]
            if len(filtered_audio_data) > 0:
                output_str = await asyncio.to_thread(handle_transcription_data, filtered_audio_data)
                if output_str and len(output_str) > 0 and nlp:
                    for sent in nlp(output_str[0]).sentences:
                        sent_id = f"event_{uuid.uuid4().hex[:12]}"
                        for token in sent.words:
                            id = f"event_{uuid.uuid4().hex[:12]}"
                            delta_resp = {
                                "event_id": id,
                                "type": "conversation.item.input_audio_transcription.delta",
                                "item_id": item_id,
                                "context_index": 0,
                                "delta": token.text,
                            }
                            try:
                                await websocket.send_json(delta_resp)
                            except Exception as e:
                                print(f"Websocket send failed: {e}")
                        sent_resp = {
                            "event_id": sent_id,
                            "type": "conversation.item.input_audio_transcription.completed",
                            "item_id": item_id,
                            "context_index": 0,
                            "transcript": sent.text,
                        }
                        try:
                            await websocket.send_json(sent_resp)
                        except Exception as e:
                            print(f"Websocket send failed: {e}")


if __name__ == "__main__":

    asr = WhisperModelCPU(
        model_name_or_path="openai/whisper-small", language="english", device="cpu", return_timestamps=True
    )

    # Test multilanguage asr
    asr.language = "chinese"
    urllib.request.urlretrieve(
        "https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/labixiaoxin.wav",
        "sample.wav",
    )
    text = asr.audio2text("sample.wav")

    asr.language = "english"
    urllib.request.urlretrieve(
        "https://github.com/intel/intel-extension-for-transformers/raw/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav",
        "sample.wav",
    )
    text = asr.audio2text("sample.wav")
    """
    # Test ASR with Intel Arc
    asr = WhisperModelXPU(
        model_name_or_path="openai/whisper-small", language="english", device="xpu", return_timestamps=True
    )

    # Test multilanguage asr
    asr.language = "chinese"
    urllib.request.urlretrieve(
        "https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/labixiaoxin.wav",
        "sample.wav",
    )
    text = asr.audio2text("sample.wav")

    asr.language = "english"
    urllib.request.urlretrieve(
        "https://github.com/intel/intel-extension-for-transformers/raw/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav",
        "sample.wav",
    )
    text = asr.audio2text("sample.wav")
    """
