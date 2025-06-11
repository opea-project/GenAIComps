# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os

# from decorators import log_latency
from dotenv import load_dotenv
from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder, OpenAIEncoder
from semantic_router.routers import SemanticRouter

from comps.cores.telemetry.opea_telemetry import opea_telemetry
from comps.router.src.integrations.controllers.base_controller import BaseController

load_dotenv()
hf_token = os.getenv("HF_TOKEN", "")
openai_api_key = os.getenv("OPENAI_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class SemanticRouterController(BaseController):
    def __init__(self, config, api_key=None, model_map=None):
        self.config = config
        self.model_map = model_map or {}

        # grab provider + model mapping
        provider = config.get("embedding_provider", "").lower()
        models = config.get("embedding_models", {})

        if provider not in {"huggingface", "openai"}:
            raise ValueError(f"Unsupported embedding_provider: '{provider}'")
        if provider not in models:
            raise ValueError(f"No embedding_models entry for provider '{provider}'")

        model_name = models[provider]
        logging.info(f"SemanticRouter using {provider} encoder '{model_name}'")

        if provider == "huggingface":
            hf_token = os.getenv("HF_TOKEN", "")
            self.encoder = HuggingFaceEncoder(
                name=model_name,
                model_kwargs={"token": hf_token},
                tokenizer_kwargs={"token": hf_token},
            )
        else:
            if not api_key:
                raise ValueError("valid api key is required for selected model provider")
            os.environ["OPENAI_API_KEY"] = api_key
            self.encoder = OpenAIEncoder(model=model_name)

        # build your routing layer
        self._build_route_layer()

    def _build_route_layer(self):
        # Build routes from the local controller config
        routes = self.config.get("routes", [])
        route_list = [Route(name=route["name"], utterances=route["utterances"]) for route in routes]

        # Reinitialize SemanticRouter to clear previous embeddings when switching models
        self.route_layer = SemanticRouter(encoder=self.encoder, routes=route_list)
        logging.info("[DEBUG] Successfully re-initialized SemanticRouter with fresh embeddings.")

    @opea_telemetry
    def route(self, messages):
        """Determines which inference endpoint to use based on the provided messages.

        It looks up the model_map to retrieve the nested endpoint value.
        """
        query = messages[0]["content"]

        route_choice = self.route_layer(query)
        endpoint_key = route_choice.name

        if not endpoint_key:
            routes = self.config.get("routes", [])
            if routes:
                endpoint_key = routes[0]["name"]
            else:
                raise ValueError("No routes available in the configuration.")

        # Lookup the endpoint in the model_map
        model_entry = self.model_map.get(endpoint_key)
        if model_entry is None:
            raise ValueError(f"Inference endpoint '{endpoint_key}' not found in global model_map.")

        # Return the endpoint from the model map
        return model_entry["endpoint"]
