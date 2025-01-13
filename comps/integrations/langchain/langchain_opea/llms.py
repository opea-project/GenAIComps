# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations  # type: ignore[import-not-found]

from typing import Any, Dict, List, Optional

import openai
from langchain_core.utils import secret_from_env
from langchain_openai.llms.base import BaseOpenAI
from pydantic import Field, SecretStr, model_validator
from typing_extensions import Self

DEFAULT_MODEL_ID = "Intel/neural-chat-7b-v3-3"


class OPEALLM(BaseOpenAI):  # type: ignore[override]
    """OPEA OPENAI Compatible LLM Endpoints.

    OPEALLM is a class to interact with OPEA OpenAI compatible large
    language model endpoints.

    To use, you should have the environment variable ``opea_api_key`` set
    with your API token, or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.llms.opea import OPEALLM

            llm = OPEALLM(
                model="Intel/neural-chat-7b-v3-3",
                max_tokens=200,
                presence_penalty=0,
                temperature=0.1,
                top_p=0.9,
            )
    """

    model_name: str = Field(alias="model", default=DEFAULT_MODEL_ID)
    """Model name to use."""
    opea_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("OPEA_API_KEY", default=None),
    )
    """OPEA_API_KEY.

    Automatically read from env variable `OPEA_API_KEY` if not provided.
    """
    opea_api_base: str = Field(default="http://localhost:9009/v1/")
    """Base URL path for API requests."""

    top_p: float = 0.6

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "opea"]

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"opea_api_key": "OPEA_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.streaming and self.n > 1:
            raise ValueError("Cannot stream results when n > 1.")
        if self.streaming and self.best_of > 1:
            raise ValueError("Cannot stream results when best_of > 1.")
        client_params: dict = {
            "api_key": self.opea_api_key.get_secret_value() if self.opea_api_key else None,
            "base_url": self.opea_api_base,
        }

        if client_params["api_key"] is None:
            raise ValueError(
                "OPEA_API_KEY is not set. Please set it in the `opea_api_key` field or "
                "in the `OPEA_API_KEY` environment variable."
            )

        if not self.client:
            sync_specific = {"http_client": self.http_client}
            self.client = openai.OpenAI(**client_params, **sync_specific).completions  # type: ignore[arg-type]
        if not self.async_client:
            async_specific = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,  # type: ignore[arg-type]
            ).completions

        return self

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        openai_params = {"model": self.model_name}
        return {**openai_params, **super()._invocation_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "opea"
