import logging
from controllers.base_controller import BaseController
from routellm.controller import Controller as RouteLLM_Controller
from decorators import log_latency
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

class RouteLLMController(BaseController):
    def __init__(self, config, hf_token=None, api_key=None, model_map=None):
        self.config = config
        # Use the provided model_map or default to an empty dict
        self.model_map = model_map or {}
        self.threshold = config.get("threshold", 0.2)
        self.routing_algorithm = config.get("routing_algorithm")
        self.controller_type = config.get("controller_type")

        # Extract nested model info from model_map:
        strong_model_info = self.model_map.get("strong", {})
        weak_model_info = self.model_map.get("weak", {})

        # Retrieve the model IDs from the nested structure
        strong_model_name = strong_model_info.get("model_id")
        weak_model_name = weak_model_info.get("model_id")

        # Create the underlying RouteLLM_Controller with the unwrapped model names.
        self.controller = RouteLLM_Controller(
            routers=[self.routing_algorithm],
            strong_model=strong_model_name,
            weak_model=weak_model_name,
            hf_token=hf_token,
            api_key=api_key,
            config=config.get("config"),
        )

    @log_latency
    def route(self, messages):

        # Get the routed model name (model_id) from the underlying RouteLLM_Controller.
        routed_model_name = self.controller.get_routed_model(
            messages,
            router=self.routing_algorithm,
            threshold=self.threshold,
        )

        # Look up the matching endpoint by comparing model_id.
        endpoint_key = next(
            (
                key
                for key, value in self.model_map.items()
                if value.get("model_id") == routed_model_name
            ),
            None,
        )

        if not endpoint_key:
            raise ValueError(f"Routed model '{routed_model_name}' not found in the model_map.")

        endpoint = self.model_map[endpoint_key]["endpoint"]

        return endpoint
