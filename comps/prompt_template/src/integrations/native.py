# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Union, List, Set

from comps import (
    CustomLogger,
    LLMParamsDoc,
    TextDoc,
    PromptTemplateInput,
    OpeaComponent,
    OpeaComponentRegistry,
    ServiceType,
)

from comps.prompt_template.src.integrations.utils.templates import (
    template_system_english as default_system_template,
    template_user_english as default_user_template,
)

from comps.prompt_template.src.integrations.utils.conversation_history_handler import ConversationHistoryHandler

logger = CustomLogger("opea_prompt_template")


@OpeaComponentRegistry.register("OPEA_PROMPT_TEMPLATE")
class OPEAPromptTemplateGenerator(OpeaComponent):
    def __init__(self, name: str, description: str, config: dict = {}):
        super().__init__(name, ServiceType.PROMPT_TEMPLATE.name.lower(), description, config)
        self._if_conv_history_in_prompt: bool = False
        self._conversation_history_placeholder: str = "conversation_history"
        self.ch_handler = ConversationHistoryHandler()

        try:
            self._validate(default_system_template, default_user_template)
            self.system_prompt_template = default_system_template
            self.user_prompt_template = default_user_template
        except ValueError as e:
            logger.error(f"Default prompt template validation failed, err={e}")
            raise

        logger.info("OPEAPromptTemplateGenerator initialized with default templates.")

    def _validate(
        self,
        system_prompt_template: str,
        user_prompt_template: str,
        placeholders: Set[str] = {"user_prompt", "reranked_docs"},
    ) -> None:
        """
        Validate system and user prompt templates.

        Args:
            system_prompt_template: System prompt template string.
            user_prompt_template: User prompt template string.
            placeholders: Required placeholders set.

        Raises:
            ValueError: For various validation failures.
        """
        if not system_prompt_template.strip() or not user_prompt_template.strip():
            raise ValueError("Prompt templates cannot be empty.")

        system_placeholders = extract_placeholders_from_template(system_prompt_template)
        user_placeholders = extract_placeholders_from_template(user_prompt_template)

        if not system_placeholders and not user_placeholders:
            raise ValueError("Prompt templates do not contain any placeholders.")

        if not placeholders:
            raise ValueError("Expected placeholders set cannot be empty.")

        duplicates = system_placeholders.intersection(user_placeholders)
        if duplicates:
            raise ValueError(f"System and user prompt templates share placeholders: {duplicates}")

        combined_placeholders = system_placeholders.union(user_placeholders)

        missing = placeholders - combined_placeholders
        if missing:
            raise ValueError(f"Prompt templates missing required placeholders: {missing}")

        extras = combined_placeholders - placeholders
        extras_no_conv = extras - {self._conversation_history_placeholder}
        if extras_no_conv:
            raise ValueError(f"Prompt templates contain unexpected placeholders: {extras_no_conv}")

        if self._conversation_history_placeholder in extras:
            self._if_conv_history_in_prompt = True
        else:
            logger.warning(
                "Placeholder {conversation_history} missing. LLM will not remember previous answers."
                " Add {conversation_history} placeholder if conversation history is desired."
            )
            self._if_conv_history_in_prompt = False

    def _changed(
        self,
        new_system_prompt_template: str,
        new_user_prompt_template: str,
        placeholders: Set[str],
    ) -> bool:
        """
        Check if new templates differ and validate them.

        Args:
            new_system_prompt_template: New system prompt template.
            new_user_prompt_template: New user prompt template.
            placeholders: Expected placeholders set.

        Returns:
            True if templates changed and valid, False otherwise.
        """
        if not new_system_prompt_template.strip() and not new_user_prompt_template.strip():
            logger.info("Empty new prompt templates, no change.")
            return False

        if (
            new_system_prompt_template == getattr(self, "system_prompt_template", None)
            and new_user_prompt_template == getattr(self, "user_prompt_template", None)
        ):
            logger.info("Prompt templates unchanged.")
            return False

        self._validate(new_system_prompt_template, new_user_prompt_template, placeholders)
        self.system_prompt_template = new_system_prompt_template
        self.user_prompt_template = new_user_prompt_template
        return True

    def _get_prompt(self, **kwargs) -> tuple[str, str]:
        """
        Generate formatted prompts with provided kwargs.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.system_prompt_template.format(**kwargs).strip()
        user_prompt = self.user_prompt_template.format(**kwargs).strip()
        return system_prompt, user_prompt

    def _parse_reranked_docs(self, reranked_docs: List[Union[dict, TextDoc]]) -> str:
        """
        Format reranked documents into string.

        Args:
            reranked_docs: List of document dicts or TextDoc instances.

        Returns:
            Formatted string with sources and sections.
        """
        formatted_docs = []
        for doc in reranked_docs:
            metadata = None
            text = None

            if isinstance(doc, dict):
                metadata = doc.get("metadata")
                text = doc.get("text")
            elif isinstance(doc, TextDoc):
                metadata = getattr(doc, "metadata", None)
                text = getattr(doc, "text", None)
            else:
                logger.error(f"Unsupported document type: {type(doc)}")
                raise ValueError(f"Unsupported document type: {type(doc)}")

            if not metadata and not text:
                logger.error(f"Document {doc} lacks metadata and text.")
                raise ValueError(f"Document {doc} lacks metadata and text.")

            file_info = "Unknown Source"
            if metadata:
                file_info = metadata.get("url") or metadata.get("object_name") or file_info

            headers = [metadata.get(f"Header{i}") for i in range(1, 8) if metadata and metadata.get(f"Header{i}")]
            header_part = f" | Section: {' > '.join(headers)}" if headers else ""

            formatted_docs.append(f"[File: {file_info}{header_part}]\n{text or ''}")

        return "\n\n".join(formatted_docs)

    async def invoke(self, input: PromptTemplateInput) -> LLMParamsDoc:
        """
        Entry point for prompt generation.

        Args:
            input: PromptTemplateInput instance.

        Returns:
            LLMParamsDoc with combined system and user prompts.
        """
        keys = set(input.data.keys())
        logger.debug(f"Input data keys: {keys}")

        if input.system_prompt_template and input.user_prompt_template:
            if self._changed(input.system_prompt_template, input.user_prompt_template, keys):
                logger.info("Prompt templates updated.")
                logger.debug(f"System template:\n{self.system_prompt_template}")
                logger.debug(f"User template:\n{self.user_prompt_template}")
            else:
                logger.debug("Prompt templates not updated.")
                expected_sys = extract_placeholders_from_template(self.system_prompt_template)
                expected_user = extract_placeholders_from_template(self.user_prompt_template)
                expected = expected_sys.union(expected_user) - {self._conversation_history_placeholder}
                if keys != expected:
                    logger.error(f"Input keys {keys} do not match expected {expected}")
                    raise ValueError(f"Input keys {keys} do not match expected {expected}")

        prompt_data = {}
        for k, v in input.data.items():
            if k == "reranked_docs":
                prompt_data[k] = self._parse_reranked_docs(v)
            else:
                prompt_data[k] = extract_text_from_nested_dict(v)
            logger.debug(f"Extracted text for key '{k}': {prompt_data[k]}")

        if self._if_conv_history_in_prompt:
            params = {}
            prompt_data[self._conversation_history_placeholder] = self.ch_handler.parse_conversation_history(
                input.conversation_history, input.conversation_history_parse_type, params
            )

        try:
            system_prompt, user_prompt = self._get_prompt(**prompt_data)
        except KeyError as e:
            logger.error(f"Missing key in prompt data: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            raise

        combined_chat_template = system_prompt + "\n" + user_prompt
        return LLMParamsDoc(chat_template=combined_chat_template, query=user_prompt)


    def check_health(self) -> bool:
        """
        Check the health status of the prompt template component.

        Returns:
            bool: True if healthy, False otherwise.
        """
        try:
            if not self.system_prompt_template or not self.user_prompt_template:
                logger.error("System or user prompt template is empty.")
                return False
            self._validate(self.system_prompt_template, self.user_prompt_template)
            return True
        except Exception as e:
            logger.error(f"Prompt template health check failed: {e}")
            return False

def extract_placeholders_from_template(template: str) -> Set[str]:
    """
    Extract placeholders from a template string.

    Args:
        template: Template string.

    Returns:
        Set of placeholder names.
    """
    return set(re.findall(r"\{(\w+)\}", template))


def extract_text_from_nested_dict(data: object) -> str:
    """
    Recursively extract text from nested dicts/lists/TextDoc.

    Args:
        data: Input data.

    Returns:
        Extracted text string.
    """
    if isinstance(data, str):
        return data
    elif data is None:
        return ""
    elif isinstance(data, TextDoc):
        return data.text
    elif isinstance(data, list):
        return " ".join(extract_text_from_nested_dict(item) for item in data)
    elif isinstance(data, dict):
        return " ".join(extract_text_from_nested_dict(v) for v in data.values())
    else:
        logger.error(f"Unsupported data type for text extraction: {type(data)}")
        raise ValueError(f"Unsupported data type for text extraction: {type(data)}")
