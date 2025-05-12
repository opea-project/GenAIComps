import logging
import os
from comps.router.src.integrations.controllers.base_controller import BaseController
from routellm.controller import Controller as RouteLLM_Controller

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

class RouteLLMController(BaseController):
    def __init__(self, config, hf_token=None, api_key=None, model_map=None):
        self.config = config
        self.model_map = model_map or {}

        # TODO: Double check this 
        default_embed = config.get("embedding_model_name")
        self.embedding_model = os.getenv(
            "ROUTELLM_EMBEDDING_MODEL_NAME",
            default_embed
        )
        logging.info(f"[RouteLLM] using embedding model: {self.embedding_model}")

        self.config.setdefault("config", {}) \
                   .setdefault("mf", {})["embedding_model_name"] = self.embedding_model

        self.threshold = config.get("threshold", 0.2)
        self.routing_algorithm = config.get("routing_algorithm")

        strong_model = self.model_map.get("strong", {}).get("model_id")
        weak_model   = self.model_map.get("weak",   {}).get("model_id")

        self.controller = RouteLLM_Controller(
            routers=[self.routing_algorithm],
            strong_model=strong_model,
            weak_model=weak_model,
            hf_token=hf_token,
            api_key=api_key,
            config=config.get("config"),
        )

    def route(self, messages):

        routed_name = self.controller.get_routed_model(
            messages,
            router=self.routing_algorithm,
            threshold=self.threshold,
        )

        endpoint_key = next(
            (k for k, v in self.model_map.items()
             if v.get("model_id") == routed_name),
            None
        )
        if not endpoint_key:
            raise ValueError(f"Routed model '{routed_name}' not in model_map")

        return self.model_map[endpoint_key]["endpoint"]
