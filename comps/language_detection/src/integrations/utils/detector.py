# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ftlangdetect import detect


def detect_language(input_text):
    """Detects the language of input text.

    Uses language detection model to get detected language.
    """
    input_text = input_text.replace("\n", " ")
    detection = detect(text=input_text, low_memory=True)
    detected_lang = detection["lang"]

    return detected_lang
