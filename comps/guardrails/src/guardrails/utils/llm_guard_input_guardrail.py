# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import HTTPException
from llm_guard import scan_prompt
from utils.llm_guard_input_scanners import InputScannersConfig

from comps import LLMParamsDoc, get_opea_logger

logger = get_opea_logger("opea_llm_guard_input_guardrail_microservice")


class OPEALLMGuardInputGuardrail:
    """OPEALLMGuardInputGuardrail is responsible for scanning and sanitizing LLM input prompts
    using various input scanners provided by LLM Guard.

    This class initializes the input scanners based on the provided configuration and
    scans the input prompts to ensure they meet the required guardrail criteria.

    Attributes:
        _scanners (list): A list of enabled scanners.

    Methods:
        __init__(usv_config: dict):
            Initializes the OPEALLMGuardInputGuardrail with the provided configuration.

        scan_llm_input(input_doc: LLMParamsDoc) -> tuple[str, dict[str, bool], dict[str, float]]:
            Scans the prompt from an LLMParamsDoc object and returns the sanitized prompt,
            validation results, and scores.
    """

    def __init__(self, usv_config: dict):
        """Initializes the OPEALLMGuardInputGuardrail with the provided configuration.

        Args:
            usv_config (dict): The configuration dictionary for initializing the input scanners.

        Raises:
            Exception: If an unexpected error occurs during the initialization of scanners.
        """
        try:
            self._scanners_config = InputScannersConfig(usv_config)
            self._scanners = self._scanners_config.create_enabled_input_scanners()
        except ValueError as e:
            logger.exception(f"Value Error occurred while initializing LLM Guard Input Guardrail scanners: {e}")
            raise
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during initializing \
                    LLM Guard Input Guardrail scanners: {e}"
            )
            raise

    def _get_anonymize_vault(self):
        anon = [item for item in self._scanners if type(item).__name__ == "Anonymize"]
        if len(anon) > 0:
            return anon[0]._vault.get()
        return None

    def _recreate_anonymize_scanner_if_exists(self):
        anon = [item for item in self._scanners if type(item).__name__ == "Anonymize"]
        if len(anon) > 0:
            logger.info(f"Anonymize scanner found: {len(anon)}")
            self._scanners.remove(anon[0])
            self._scanners.append(self._scanners_config._create_anonymize_scanner())

    def _analyze_scan_outputs(self, prompt, results_valid, results_score):
        filtered_results_valid_no_redacted = {}
        scanners_with_redact = ["BanCompetitors", "BanSubstrings", "OPEABanSubstrings", "Regex", "OPEARegexScanner"]

        for key, value in results_valid.items():
            if_redacted = False
            redacted_scanner = [
                item
                for item in self._scanners
                if type(item).__name__ in scanners_with_redact and type(item).__name__ == key
            ]

            if len(redacted_scanner) > 0:
                if_redacted = redacted_scanner[0]._redact

            if key != "Anonymize" and not if_redacted:
                filtered_results_valid_no_redacted[key] = value

        if False in filtered_results_valid_no_redacted.values():
            msg = f"Prompt {prompt} is not valid, scores: {results_score}"
            logger.error(f"{msg}")
            usr_msg = "I'm sorry, I cannot assist you with your prompt."
            raise HTTPException(status_code=466, detail=f"{usr_msg}")

    def scan_llm_input(self, input_doc: LLMParamsDoc) -> LLMParamsDoc:
        """Scan the prompt from an LLMParamsDoc object.

        Args:
            input_doc (LLMParamsDoc): The input document containing the prompt to be scanned.

        Returns:
            tuple[str, dict[str, bool], dict[str, float]]: A tuple containing the sanitized prompt,
            a dictionary of validation results, and a dictionary of scores.

        Raises:
            HTTPException: If the prompt is not valid based on the scanner results.
        """
        fresh_scanners = False
        if input_doc.input_guardrail_params is not None:
            if self._scanners_config.changed(input_doc.input_guardrail_params.dict()):
                self._scanners = self._scanners_config.create_enabled_input_scanners()
                fresh_scanners = True
        else:
            logger.warning("Input guardrail params not found in input document.")
        if self._scanners:
            if not fresh_scanners:
                logger.info("Recreating anonymize scanner if exists to clear the Vault.")
                self._recreate_anonymize_scanner_if_exists()
            system_prompt = input_doc.messages.system
            user_prompt = input_doc.messages.user

            # We want to block only user question with a TokenLimit Scanner
            scanners_without_token_limit = [item for item in self._scanners if type(item).__name__ != "TokenLimit"]
            if len(self._scanners) != scanners_without_token_limit:
                sanitized_system_prompt, system_results_valid, system_results_score = scan_prompt(
                    scanners_without_token_limit, system_prompt
                )
            else:
                sanitized_system_prompt, system_results_valid, system_results_score = scan_prompt(
                    self._scanners, system_prompt
                )

            if "### Question:" in user_prompt:
                # Default template is used
                prefix = "### Question: "
                suffix = " \n ### Answer:"
                user_prompt_to_scan = user_prompt.split(prefix)[1].split(suffix)[0]
                sanitized_user_prompt, user_results_valid, user_results_score = scan_prompt(
                    self._scanners, user_prompt_to_scan
                )
                sanitized_user_prompt = prefix + sanitized_user_prompt + suffix
            else:
                sanitized_user_prompt, user_results_valid, user_results_score = scan_prompt(self._scanners, user_prompt)

            self._analyze_scan_outputs(system_prompt, system_results_valid, system_results_score)
            self._analyze_scan_outputs(user_prompt, user_results_valid, user_results_score)

            input_doc.messages.system = sanitized_system_prompt
            input_doc.messages.user = sanitized_user_prompt
            if input_doc.output_guardrail_params is not None and "Anonymize" in user_results_valid:
                input_doc.output_guardrail_params.anonymize_vault = self._get_anonymize_vault()
            elif input_doc.output_guardrail_params is None and "Anonymize" in user_results_valid:
                logger.warning("No output guardrails params, could not append the vault for Anonymize scanner.")
            return input_doc
        else:
            logger.info("No input scanners enabled. Skipping scanning.")
            return input_doc
