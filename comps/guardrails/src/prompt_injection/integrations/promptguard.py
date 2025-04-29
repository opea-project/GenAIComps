# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

from transformers import pipeline

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType, TextDoc

logger = CustomLogger("opea_prompt_guard_promptguard")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("NATIVE_PROMPT_INJECTION_DETECTION")
class OpeaPromptInjectionPromptGuard(OpeaComponent):
    """A specialized prompt injection component derived from OpeaComponent."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.GUARDRAIL.name.lower(), description, config)
        self.hf_token = os.getenv("HF_TOKEN")
        self.model = os.getenv("PROMPT_INJECTION_DETECTION_MODEL", "meta-llama/Prompt-Guard-86M")
        self.pi_pipeline = pipeline("text-classification", model=self.model, tokenizer=self.model)
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaPromptInjectionNative health check failed.")

    async def invoke(self, input: TextDoc):
        """Invokes the prompt injection/jailbreak detection for the input.

        Args:
            input (Input TextDoc)
        """
        result = await asyncio.to_thread(self.pi_pipeline, input.text)

        if result[0]["label"].lower() == "jailbreak":
            return TextDoc(text="Violated policies: jailbreak, please check your input.", downstream_black_list=[".*"])
        elif result[0]["label"].lower() == "injection":
            return TextDoc(
                text="Violated policies: prompt injection, please check your input.", downstream_black_list=[".*"]
            )
        else:
            return TextDoc(text=input.text)

    def check_health(self) -> bool:
        """Checks the health of the prompt injection service using Prompt Guard.

        Returns:
            bool: True if service is reachable and healthy, False otherwise.
        """
        try:
            self.pi_pipeline
            return True

        except Exception as e:
            logger.error(f"Health check failed due to an exception: {e}")
            return False
