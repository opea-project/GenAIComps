# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from funasr import AutoModel

logger = logging.getLogger(__name__)

FUNASR_MODEL_MAP = {
    "paraformer-zh": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "paraformer-en": "iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
    "paraformer-online": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
}


class Paraformer:
    def __init__(self, model_name, device="cpu", revision="v2.0.4"):
        if model_name not in FUNASR_MODEL_MAP:
            raise ValueError(
                f"Invalid ASR model name {model_name}. Supported models are: {list(FUNASR_MODEL_MAP.keys())}"
            )

        self.model_name = model_name
        model_name = FUNASR_MODEL_MAP[model_name]
        # use same vad and punc model for different ASR models
        self.model = AutoModel(
            model=model_name,
            model_revision=revision,
            vad_model="fsmn-vad",
            vad_model_revision="v2.0.4",
            punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            punc_model_revision="v2.0.4",
            #   spk_model="cam++", spk_model_revision="v2.0.2",
            device=device,
            disable_update=True,
        )

    def transcribe(self, audio_path: str) -> str:
        try:
            res = self.model.generate(input=audio_path)
            # res [{'key': <input>, 'text': '...', , 'timestamp': [[], [], ...]}]
            if len(res) > 0:
                return res[0]["text"]
            else:
                logger.error("ASR transcription generated empty result.")
                return None
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return None
