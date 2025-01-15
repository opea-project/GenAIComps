# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

import openai
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.utils import secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

DEFAULT_MODEL_ID = "Intel/neural-chat-7b-v3-3"


class ChatOPEA(BaseChatOpenAI):  # type: ignore[override]
    """OPEA OPENAI Compatible Chat large language models.

    See https://opea.dev/ for information about OPEA.

    To use, you should have the ``openai`` python package installed and the
    environment variable ``opea_api_key`` set with your API token.
    Alternatively, you can use the opea_api_key keyword argument.

    Any parameters that are valid to be passed to the `openai.create` call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatOPEA

            chat = ChatOPEA(model_name="Intel/neural-chat-7b-v3-3")
    """  # noqa: E501

    model_name: str = Field(alias="model", default=DEFAULT_MODEL_ID)
    """Model name to use."""
    opea_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("OPEA_API_KEY", default=None),
    )
    """OPEA_API_KEY.

    Automatically read from env variable `OPEA_API_KEY` if not provided.
    """
    opea_api_base: str = Field(default="https://localhost:9009/v1/")
    """Base URL path for API requests."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example,
            {"opea_api_key": "OPEA_API_KEY"}
        """
        return {"opea_api_key": "OPEA_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "opea"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        """List of attribute names that should be included in the serialized kwargs.

        These attributes must be accepted by the constructor.
        """
        attributes: Dict[str, Any] = {}

        if self.opea_api_base:
            attributes["opea_api_base"] = self.opea_api_base
        return attributes

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "opea-chat"

    def _get_ls_params(self, stop: Optional[List[str]] = None, **kwargs: Any) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "opea"
        return params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        client_params: dict = {
            "api_key": (self.opea_api_key.get_secret_value() if self.opea_api_key else None),
            "base_url": self.opea_api_base,
        }

        if client_params["api_key"] is None:
            raise ValueError(
                "OPEA_API_KEY is not set. Please set it in the `opea_api_key` field or "
                "in the `OPEA_API_KEY` environment variable."
            )

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.client = openai.OpenAI(**client_params, **sync_specific).chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(**client_params, **async_specific).chat.completions
        return self

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        openai_params = {"model": self.model_name}
        return {**openai_params, **super()._invocation_params}
