# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from routellm.controller import Controller as RouteLLM_Controller

from comps.router.src.integrations.controllers.base_controller import BaseController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class RouteLLMController(BaseController):
    def __init__(self, config, hf_token=None, api_key=None, model_map=None):
        self.config = config
        self.model_map = model_map or {}

        # Determine embedding provider
        provider = config.get("embedding_provider", "huggingface").lower()

        # Resolve embedding model: env override ↔️ config default
        env_var = "ROUTELLM_EMBEDDING_MODEL_NAME"
        default_model = config.get("embedding_model_name")
        self.embedding_model = os.getenv(env_var, default_model)
        if not self.embedding_model:
            raise ValueError(f"No embedding_model_name in config and {env_var} not set")
        logging.info(f"[RouteLLM] using {provider} embedding model: {self.embedding_model}")

        # Inject into nested mf config
        nested = self.config.setdefault("config", {})
        mf = nested.setdefault("mf", {})
        mf["embedding_model_name"] = self.embedding_model

        # Validate routing settings
        self.routing_algorithm = config.get("routing_algorithm")
        if not self.routing_algorithm:
            raise ValueError("routing_algorithm must be specified in configuration")
        self.threshold = config.get("threshold", 0.2)

        # Extract strong/weak model IDs
        strong_model = self.model_map.get("strong", {}).get("model_id")
        weak_model = self.model_map.get("weak", {}).get("model_id")
        if not strong_model or not weak_model:
            raise ValueError("model_map must include both 'strong' and 'weak' entries")

        # Prepare Env for OpenAI if needed
        if provider == "openai":
            if not api_key:
                raise ValueError("api_key is required for OpenAI embeddings")
            os.environ["OPENAI_API_KEY"] = api_key

        # Initialize the underlying controller (keyword args to match signature)
        self.controller = RouteLLM_Controller(
            routers=[self.routing_algorithm],
            strong_model=strong_model,
            weak_model=weak_model,
            config=nested,
            hf_token=hf_token if provider == "huggingface" else None,
            api_key=api_key if provider == "openai" else None,
        )

    def route(self, messages):
        routed_name = self.controller.get_routed_model(
            messages,
            router=self.routing_algorithm,
            threshold=self.threshold,
        )
        endpoint_key = next((k for k, v in self.model_map.items() if v.get("model_id") == routed_name), None)
        if not endpoint_key:
            raise ValueError(f"Routed model '{routed_name}' not in model_map")
        return self.model_map[endpoint_key]["endpoint"]
