# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class ServiceRoleType(Enum):
    """The enum of a service role."""

    MICROSERVICE = 0
    MEGASERVICE = 1


class ServiceType(Enum):
    """The enum of a service type."""

    GATEWAY = 0
    EMBEDDING = 1
    RETRIEVER = 2
    RERANK = 3
    LLM = 4
    ASR = 5
    TTS = 6
    GUARDRAIL = 7
    VECTORSTORE = 8
    DATAPREP = 9
    UNDEFINED = 10


class MegaServiceEndpoint(Enum):
    """The enum of an MegaService endpoint."""

    # OPEA Exclusive
    CHAT_QNA = "/v1/chatqna"
    AUDIO_QNA = "/v1/audioqna"
    VISUAL_QNA = "/v1/visualqna"
    CODE_GEN = "/v1/codegen"
    CODE_TRANS = "/v1/codetrans"
    DOC_SUMMARY = "/v1/docsum"
    SEARCH_QNA = "/v1/searchqna"
    TRANSLATION = "/v1/translation"
    # Follow OPENAI
    EMBEDDINGS = "/v1/embeddings"
    TTS = "/v1/audio/speech"
    ASR = "/v1/audio/transcriptions"
    CHAT = "/v1/chat/completions"
    RETRIEVAL = "/v1/retrieval"
    RERANKING = "/v1/reranking"
    GUARDRAILS = "/v1/guardrails"
    # COMMON
    LIST_SERVICE = "/v1/list_service"
    LIST_PARAMETERS = "/v1/list_parameters"

    def __str__(self):
        return self.value


class MicroServiceEndpoint(Enum):
    """The enum of an MicroService endpoint."""

    EMBEDDINGS = "/v1/microservice/embeddings"
    TTS = "/v1/microservice/tts"
    ASR = "/v1/microservice/asr"
    CHAT = "/v1/microservice/chat"
    RETRIEVAL = "/v1/microservice/retrieval"
    RERANKING = "/v1/microservice/reranking"
    GUARDRAILS = "/v1/microservice/guardrails"

    def __str__(self):
        return self.value
