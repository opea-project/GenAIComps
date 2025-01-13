# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test chat model integration."""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_opea.chat_models import ChatOPEA
from langchain_tests.integration_tests import ChatModelIntegrationTests

OPEA_API_BASE = "http://localhost:9009/v1"
OPEA_API_KEY = "my_secret_value"
MODEL_NAME = "Intel/neural-chat-7b-v3-3"


class TestChatOPEA(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatOPEA]:
        return ChatOPEA

    @property
    def chat_model_params(self) -> dict:
        return {
            "opea_api_base": OPEA_API_BASE,
            "opea_api_key": OPEA_API_KEY,
            "model_name": MODEL_NAME,
        }

    @property
    def supports_image_inputs(self) -> bool:
        return False

    @property
    def has_tool_calling(self) -> bool:
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        return False

    @property
    def supports_anthropic_inputs(self) -> bool:
        return False

    supports_anthropic_inputs

    @pytest.mark.xfail(reason=("Fails with 'AssertionError'. OPEA does not support 'tool_choice' yet."))
    def test_structured_output(self, model: BaseChatModel) -> None:
        super().test_structured_output(model)

    @pytest.mark.xfail(reason=("Fails with 'AssertionError'. OPEA does not support 'tool_choice' yet."))
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)
