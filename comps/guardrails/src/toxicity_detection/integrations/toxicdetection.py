# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

from transformers import pipeline

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType, TextDoc

logger = CustomLogger("opea_toxicity_native")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("OPEA_NATIVE_TOXICITY")
class OpeaToxicityDetectionNative(OpeaComponent):
    """A specialized toxicity detection component derived from OpeaComponent."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.GUARDRAIL.name.lower(), description, config)
        self.model = os.getenv("TOXICITY_DETECTION_MODEL", "Intel/toxic-prompt-roberta")
        self.toxicity_pipeline = pipeline("text-classification", model=self.model, tokenizer=self.model)
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaToxicityDetectionNative health check failed.")

    async def invoke(self, input: TextDoc):
        """Invokes the toxic detection for the input.

        Args:
            input (Input TextDoc)
        """
        toxic = await asyncio.to_thread(self.toxicity_pipeline, input.text)
        if toxic[0]["label"].lower() == "toxic":
            return TextDoc(text="Violated policies: toxicity, please check your input.", downstream_black_list=[".*"])
        else:
            return TextDoc(text=input.text)

    def check_health(self) -> bool:
        """Checks the health of the animation service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if self.toxicity_pipeline:
            return True
        else:
            return False
