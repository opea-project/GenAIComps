#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Document
from comps.cores.proto.docarray import (
    Audio2TextDoc,
    Base64ByteStrDoc,
    DocPath,
    EmbedDoc,
    GeneratedDoc,
    LLMParamsDoc,
    SearchedDoc,
    SearchedMultimodalDoc,
    LVMSearchedMultimodalDoc,
    RerankedDoc,
    TextDoc,
    MetadataTextDoc,
    RAGASParams,
    RAGASScores,
    GraphDoc,
    LVMDoc,
    LVMVideoDoc,
    ImagePath,
    ImagesPath,
    VideoPath,
    ImageDoc,
    SDInputs,
    SDImg2ImgInputs,
    SDOutputs,
    TextImageDoc,
    MultimodalDoc,
    EmbedMultimodalDoc,
    FactualityDoc,
    ScoreDoc,
    PIIRequestDoc,
    PIIResponseDoc,
    Audio2text,
    DocSumDoc,
)

# Constants
from comps.cores.mega.constants import MegaServiceEndpoint, ServiceRoleType, ServiceType

# Microservice
from comps.cores.mega.orchestrator import ServiceOrchestrator
from comps.cores.mega.orchestrator_with_yaml import ServiceOrchestratorWithYaml
from comps.cores.mega.micro_service import MicroService, register_microservice, opea_microservices

# Telemetry
from comps.cores.telemetry.opea_telemetry import opea_telemetry

# Common
from comps.cores.common.component import OpeaComponent, OpeaComponentController

# Statistics
from comps.cores.mega.base_statistics import statistics_dict, register_statistics

# Logger
from comps.cores.mega.logger import CustomLogger
