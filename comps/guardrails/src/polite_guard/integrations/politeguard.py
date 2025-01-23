# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

from transformers import pipeline

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType, TextDoc

logger = CustomLogger("opea_polite_guard")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("OPEA_POLITE_GUARD")
class OpeaPoliteGuard(OpeaComponent):
    """A specialized politeness detection component derived from OpeaComponent."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.GUARDRAIL.name.lower(), description, config)
        self.model = os.getenv("POLITE_GUARD_MODEL", "Intel/polite-guard")
        self.polite_pipeline = pipeline("text-classification", model=self.model, tokenizer=self.model)
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaPoliteGuard health check failed.")

    async def invoke(self, input: str):
        """Invokes the polite guard for the input.

        Args:
            input (Input str)
        """
        response = await asyncio.to_thread(self.polite_pipeline, input)
        if response[0]["label"] == "impolite":
            return TextDoc(text=f"Violated policies: Impolite (score: {response[0]['score']:0.2f}), please check your input.", downstream_black_list=[".*"])
        else:
            return TextDoc(text=input)

    def check_health(self) -> bool:
        """Checks the health of the animation service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if self.polite_pipeline:
            return True
        else:
            return False
