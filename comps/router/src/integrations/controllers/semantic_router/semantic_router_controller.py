# semantic_router_controller.py

import logging
import os
from controllers.base_controller import BaseController
from semantic_router.layer import RouteLayer
from semantic_router.encoders import OpenAIEncoder, HuggingFaceEncoder
from semantic_router import Route
from decorators import log_latency
from dotenv import load_dotenv

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
        self.controller_type = config.get("controller_type")
        embedding_provider = self.config.get("embedding_provider", "huggingface")

        logging.info(f"Embedding Provider: {embedding_provider}")

        if not api_key:
            raise ValueError("Please set api_key for semantic router.")

        try:
            if embedding_provider == "huggingface":
                self.encoder = HuggingFaceEncoder(
                    name="BAAI/bge-base-en-v1.5",  # Specify your Hugging Face model here
                    model_kwargs={"token": hf_token},
                    tokenizer_kwargs={"token": hf_token}
                )
            elif embedding_provider == "openai":
                os.environ["OPENAI_API_KEY"] = api_key
                self.encoder = OpenAIEncoder()
            else:
                raise ValueError(f"Unsupported encoder type: {embedding_provider}")

        except Exception as e:
            logging.error(f"Failed to initialize encoder for {embedding_provider}: {e}")
            raise

        self._build_route_layer()

    def _build_route_layer(self):
        # Build routes from the local controller config
        routes = self.config.get("routes", [])
        route_list = [Route(name=route["name"], utterances=route["utterances"]) for route in routes]

        # Reinitialize RouteLayer to clear previous embeddings when switching models
        self.route_layer = RouteLayer(encoder=self.encoder, routes=route_list)
        logging.info("[DEBUG] Successfully re-initialized RouteLayer with fresh embeddings.")

    @log_latency
    def route(self, messages):
        """
        Determines which inference endpoint to use based on the provided messages.
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
