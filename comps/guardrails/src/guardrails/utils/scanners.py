# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Iterable

from llm_guard.input_scanners import BanSubstrings, Regex
from llm_guard.input_scanners.regex import MatchType
from presidio_anonymizer.core.text_replace_builder import TextReplaceBuilder

from comps import CustomLogger

logger = CustomLogger("opea_llm_guard_utils_scanners")


# The bug is reported here: https://github.com/protectai/llm-guard/issues/210
class OPEABanSubstrings(BanSubstrings):
    def _redact_text(self, text: str, substrings: list[str]) -> str:
        redacted_text = text
        flags = 0
        if not self._case_sensitive:
            flags = re.IGNORECASE
        for s in substrings:
            regex_redacted = re.compile(re.escape(s), flags)
            redacted_text = regex_redacted.sub("[REDACTED]", redacted_text)
        return redacted_text

    def scan(self, prompt: str, output: str = None) -> tuple[str, bool, float]:
        if output is not None:
            return super().scan(output)
        return super().scan(prompt)


# LLM Guard's Regex Scanner doesn't replace all occurrences of found patterns.
# The bug is reported here: https://github.com/protectai/llm-guard/issues/229
class OPEARegexScanner(Regex):
    def scan(self, prompt: str, output: str = None) -> tuple[str, bool, float]:
        text_to_scan = ""
        if output is not None:
            text_to_scan = output
        else:
            text_to_scan = prompt

        text_replace_builder = TextReplaceBuilder(original_text=text_to_scan)
        for pattern in self._patterns:
            if self._match_type == MatchType.SEARCH:
                matches = re.finditer(pattern, text_to_scan)
            else:
                matches = self._match_type.match(pattern, text_to_scan)

            if matches is None:
                continue
            elif isinstance(matches, Iterable):
                matches = list(matches)
                if len(matches) == 0:
                    continue
            else:
                matches = [matches]

            if self._is_blocked:
                logger.warning(f"Pattern was detected in the text: {pattern}")

                if self._redact:
                    for match in reversed(matches):
                        text_replace_builder.replace_text_get_insertion_index(
                            "[REDACTED]",
                            match.start(),
                            match.end(),
                        )

                return text_replace_builder.output_text, False, 1.0

            logger.debug(f"Pattern matched the text: {pattern}")
            return text_replace_builder.output_text, True, 0.0

        if self._is_blocked:
            logger.debug("None of the patterns were found in the text")
            return text_replace_builder.output_text, True, 0.0

        logger.warning("None of the patterns matched the text")
        return text_replace_builder.output_text, False, 1.0
