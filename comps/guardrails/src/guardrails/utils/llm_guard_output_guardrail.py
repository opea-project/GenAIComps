# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import HTTPException
from llm_guard import scan_output
from utils.llm_guard_output_scanners import OutputScannersConfig

from comps import CustomLogger, GeneratedDoc

logger = CustomLogger("opea_llm_guard_output_guardrail_microservice")


class OPEALLMGuardOutputGuardrail:
    """OPEALLMGuardOutputGuardrail is responsible for scanning and sanitizing LLM output responses
    using various output scanners provided by LLM Guard.

    This class initializes the output scanners based on the provided configuration and
    scans the output responses to ensure they meet the required guardrail criteria.

    Attributes:
        _scanners (list): A list of enabled scanners.

    Methods:
        __init__(usv_config: list):
            Initializes the OPEALLMGuardOutputGuardrail with the provided configuration.

        scan_llm_output(output_doc: object) -> str:
            Scans the output from an LLM output document and returns the sanitized output.
    """

    def __init__(self, usv_config: list):
        """Initializes the OPEALLMGuardOutputGuardrail with the provided configuration.

        Args:
            usv_config (list): The configuration list for initializing the output scanners.

        Raises:
            Exception: If an unexpected error occurs during the initialization of scanners.
        """
        try:
            self._scanners_config = OutputScannersConfig(usv_config)
            self._scanners = self._scanners_config.create_enabled_output_scanners()
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during initializing \
                    LLM Guard Output Guardrail scanners: {e}"
            )
            raise

    def scan_llm_output(self, output_doc: GeneratedDoc) -> str:
        """Scans the output from an LLM output document.

        Args:
            output_doc (object): The output document containing the response to be scanned.

        Returns:
            str: The sanitized output.

        Raises:
            HTTPException: If the output is not valid based on the scanner results.
            Exception: If an unexpected error occurs during scanning.
        """
        try:
            if output_doc.output_guardrail_params is not None:
                self._scanners_config.vault = output_doc.output_guardrail_params.anonymize_vault
                if self._scanners_config.changed(output_doc.output_guardrail_params.dict()):
                    self._scanners = self._scanners_config.create_enabled_output_scanners()
            else:
                logger.warning("Output guardrail params not found in input document.")
            if self._scanners:
                sanitized_output, results_valid, results_score = scan_output(
                    self._scanners, output_doc.prompt, output_doc.text
                )
                if False in results_valid.values():
                    msg = f"LLM Output {output_doc.text} is not valid, scores: {results_score}"
                    logger.error(msg)
                    usr_msg = "I'm sorry, but the model output is not valid according to the policies."
                    redact_or_truncated = [
                        c.get("redact", False) or c.get("truncate", False)
                        for _, c in self._scanners_config._output_scanners_config.items()
                    ]  # to see if sanitized output available
                    if any(redact_or_truncated):
                        usr_msg = f"We sanitized the answer due to the guardrails policies: {sanitized_output}"
                    raise HTTPException(status_code=466, detail=usr_msg)
                return sanitized_output
            else:
                logger.warning("No output scanners enabled. Skipping scanning.")
                return output_doc.text
        except HTTPException as e:
            raise e
        except ValueError as e:
            error_msg = f"Validation Error occurred while initializing LLM Guard Output Guardrail scanners: {e}"
            logger.exception(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred during scanning prompt with LLM Guard Output Guardrail: {e}"
            logger.exception(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
