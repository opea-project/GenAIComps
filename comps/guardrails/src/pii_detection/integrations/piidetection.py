# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NerModelConfiguration, TransformersNlpEngine

from comps import (
    CustomLogger,
    OpeaComponent,
    OpeaComponentRegistry,
    PIIRequestDoc,
    PIIResponseDoc,
    ServiceType,
    TextDoc,
)


logger = CustomLogger("opea_pii_native")
logflag = os.getenv("LOGFLAG", False)


# Entity mappings between the model's and Presidio's
MAPPING = dict(
    PER="PERSON",
    LOC="LOCATION",
    ORG="ORGANIZATION",
    AGE="AGE",
    ID="ID",
    EMAIL="EMAIL",
    DATE="DATE_TIME",
    PHONE="PHONE_NUMBER",
    PERSON="PERSON",
    LOCATION="LOCATION",
    GPE="LOCATION",
    ORGANIZATION="ORGANIZATION",
    NORP="NRP",
    PATIENT="PERSON",
    STAFF="PERSON",
    HOSP="LOCATION",
    PATORG="ORGANIZATION",
    TIME="DATE_TIME",
    HCW="PERSON",
    HOSPITAL="LOCATION",
    FACILITY="LOCATION",
    VENDOR="ORGANIZATION",
)

LABELS_TO_IGNORE = ["O"]


@OpeaComponentRegistry.register("OPEA_NATIVE_PII")
class OpeaPiiDetectionNative(OpeaComponent):
    """A specialized pii detection component derived from OpeaComponent."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.GUARDRAIL.name.lower(), description, config)
        self.model = os.getenv("PII_DETECTION_MODEL", "StanfordAIMI/stanford-deidentifier-base")

        # Transformer model config
        model_config = [
            {
                "lang_code": "en",
                "model_name": {
                    "spacy": "en_core_web_sm",
                    "transformers": self.model,
                },
            }
        ]

        self.ner_model_configuration = NerModelConfiguration(
            model_to_presidio_entity_mapping=MAPPING,
            alignment_mode="expand",
            aggregation_strategy="max",
            labels_to_ignore=LABELS_TO_IGNORE,
        )

        self.transformers_nlp_engine = TransformersNlpEngine(
            models=model_config, ner_model_configuration=self.ner_model_configuration
        )

        # Transformer-based analyzer
        self.analyzer = AnalyzerEngine(
            nlp_engine=self.transformers_nlp_engine, supported_languages=["en"], log_decision_process=True
        )

        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaToxicityDetectionNative health check failed.")

    async def invoke(self, input: TextDoc) -> TextDoc:
        """Invokes the pii detection for the input.

        Args:
            input (Input TextDoc)
        """
        pii = await asyncio.to_thread(self.analyzer.analyze, input.text, "en")

        if pii:
            # convert AnalyzerResult to List[Dict]
            pii = [entity.to_dict() for entity in pii]

            # convert np.float32 to float
            for entity in pii:
                entity["score"] = float(entity["score"])

            return PIIResponseDoc(detected_pii=pii)
        else:
            return TextDoc(text=input.text)

    def check_health(self) -> bool:
        """Checks the health of the pii service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if self.analyzer:
            return True
        else:
            return False
