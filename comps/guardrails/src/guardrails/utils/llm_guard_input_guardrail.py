# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import HTTPException
from llm_guard import scan_prompt
from utils.llm_guard_input_scanners import InputScannersConfig

from comps import CustomLogger, LLMParamsDoc

logger = CustomLogger("opea_llm_guard_input_guardrail_microservice")


class OPEALLMGuardInputGuardrail:
    """OPEALLMGuardInputGuardrail is responsible for scanning and sanitizing LLM input prompts
    using various input scanners provided by LLM Guard."""

    def __init__(self, usv_config: dict):
        try:
            self._scanners_config = InputScannersConfig(usv_config)
            self._scanners = self._scanners_config.create_enabled_input_scanners()
        except ValueError as e:
            logger.exception(f"Value Error during scanner initialization: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during scanner initialization: {e}")
            raise

    def _get_anonymize_vault(self):
        for item in self._scanners:
            if type(item).__name__ == "Anonymize":
                return item._vault.get()
        return None

    def _recreate_anonymize_scanner_if_exists(self):
        for item in self._scanners:
            if type(item).__name__ == "Anonymize":
                logger.info("Recreating Anonymize scanner to clear Vault.")
                self._scanners.remove(item)
                self._scanners.append(self._scanners_config._create_anonymize_scanner())
                break

    def _analyze_scan_outputs(self, prompt, results_valid, results_score):
        filtered_results = {
            key: value
            for key, value in results_valid.items()
            if key != "Anonymize"
            and not (
                type(scanner := next((s for s in self._scanners if type(s).__name__ == key), None)).__name__
                in {"BanCompetitors", "BanSubstrings", "OPEABanSubstrings", "Regex", "OPEARegexScanner"}
                and getattr(scanner, "_redact", False)
            )
        }

        if False in filtered_results.values():
            msg = f"Prompt '{prompt}' is not valid, scores: {results_score}"
            logger.error(msg)
            raise HTTPException(status_code=466, detail="I'm sorry, I cannot assist you with your prompt.")

    def scan_llm_input(self, input_doc: LLMParamsDoc) -> LLMParamsDoc:
        fresh_scanners = False

        if input_doc.input_guardrail_params is not None:
            if self._scanners_config.changed(input_doc.input_guardrail_params.dict()):
                self._scanners = self._scanners_config.create_enabled_input_scanners()
                fresh_scanners = True
        else:
            logger.warning("Input guardrail params not found.")

        if not self._scanners:
            logger.info("No scanners enabled. Skipping input scan.")
            return input_doc

        if not fresh_scanners:
            self._recreate_anonymize_scanner_if_exists()

        user_prompt = input_doc.query
        sanitized_user_prompt, results_valid, results_score = scan_prompt(self._scanners, user_prompt)
        self._analyze_scan_outputs(user_prompt, results_valid, results_score)

        input_doc.query = sanitized_user_prompt

        if input_doc.output_guardrail_params is not None and "Anonymize" in results_valid:
            input_doc.output_guardrail_params.anonymize_vault = self._get_anonymize_vault()
        elif input_doc.output_guardrail_params is None and "Anonymize" in results_valid:
            logger.warning("Anonymize scanner result exists, but output_guardrail_params is missing.")

        return input_doc
