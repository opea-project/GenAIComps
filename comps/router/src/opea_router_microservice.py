import os
import logging
import requests
import yaml

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

# OPEA Imports
from comps import register_microservice, ServiceType, TextDoc

from comps.router.src.integrations.controllers.controller_factory import ControllerFactory


class RouteEndpointDoc(BaseModel):
    """Return object for the '/v1/route' microservice."""
    url: str = Field(..., description="URL of the chosen inference endpoint")

class LLMResponseDoc(BaseModel):
    """Return object for the '/v1/route-forward' microservice."""
    content: str = Field(..., description="Generated text from the chosen model")
    model: str   = Field(..., description="Name/ID of the model that produced the text")

class ReloadConfigOutputDoc(BaseModel):
    """Return object for the '/v1/route/reload-config' microservice."""
    status: str = Field(..., description="Status of config reload")
    new_config: Dict[str, Any] = Field(..., description="Updated config data")


CONFIG_PATH = os.getenv("CONFIG_PATH", "/app/config.yaml")

_config_data: Dict[str, Any] = {}
_controller_factory: Optional[ControllerFactory] = None
_controller = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def _load_config():
    """
    Loads the YAML config file from CONFIG_PATH and re-initializes the controller.
    """
    global _config_data, _controller_factory, _controller

    try:
        with open(CONFIG_PATH, "r") as f:
            new_data = yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")

    _config_data = new_data
    logging.info(f"[Router] Loaded config data from: {CONFIG_PATH}")

    if _controller_factory is None:
        _controller_factory = ControllerFactory()

    model_map = _config_data.get("model_map", {})
    controller_config_path = _config_data.get("controller_config_path")

    _controller = _controller_factory.factory(
        controller_config=controller_config_path,
        model_map=model_map
    )
    logging.info("[Router] Controller re-initialized successfully.")


# Attempt initial load at module import
try:
    _load_config()
except Exception as e:
    logging.error("[Router] Error loading initial config: %s", e)
    raise


@register_microservice(
    name="opea_service@router_route",
    service_type=ServiceType.LLM,  # TODO: may need to define a new ServiceType.ROUTER
    endpoint="/v1/route",
    host="0.0.0.0",
    port=6000,
    input_datatype=TextDoc,      # OPEA’s standard text input doc
    output_datatype=RouteEndpointDoc
)
def route_microservice(input: TextDoc) -> RouteEndpointDoc:
    """
    Microservice that decides which model endpoint is best for the given text input.
    Returns only the route URL (does not forward).
    """
    if not _controller:
        raise RuntimeError("Controller is not initialized — config load failed?")

    # Convert OPEA’s TextDoc into "messages" structure:
    query_content = input.text
    messages = [{"content": query_content}]

    try:
        endpoint = _controller.route(messages)
        if not endpoint:
            raise ValueError("No suitable model endpoint found.")
        return RouteEndpointDoc(url=endpoint)

    except Exception as e:
        logging.error("[Router] Error during model routing: %s", e)
        raise


@register_microservice(
    name="opea_service@router_forward",
    service_type=ServiceType.LLM,
    endpoint="/v1/route-forward",
    host="0.0.0.0",
    port=6000,
    input_datatype=TextDoc,
    output_datatype=LLMResponseDoc
)
def route_forward_microservice(input: TextDoc) -> LLMResponseDoc:
    """
    Microservice that:
    1) Chooses the best model endpoint for the input
    2) Forwards the query to that endpoint
    3) Returns the generated text + model name
    """
    if not _controller:
        raise RuntimeError("Controller is not initialized — config load failed?")

    query_content = input.text
    messages = [{"content": query_content}]

    try:
        endpoint = _controller.route(messages)
        if not endpoint:
            raise ValueError("No suitable model endpoint found.")

        response = requests.post(endpoint, json={"content": query_content}, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Inference service error: HTTP {response.status_code}")

        response_data = response.json()
        generated_text = response_data["choices"][0]["message"]["content"].strip()
        model = response_data.get("model", "unknown-model")

        return LLMResponseDoc(content=generated_text, model=model)

    except Exception as e:
        logging.error("[Router] route_forward error: %s", e)
        raise


@register_microservice(
    name="opea_service@router_reload_config",
    service_type=ServiceType.LLM,
    endpoint="/v1/route/reload-config",
    host="0.0.0.0",
    port=6000,
    input_datatype=TextDoc, 
    output_datatype=ReloadConfigOutputDoc
)
def route_reload_config_microservice(_: TextDoc) -> ReloadConfigOutputDoc:
    """
    Microservice that triggers a config reload. 
    We ignore the input doc's text, but we must supply *some* input doc type.
    """
    try:
        _load_config()
        return ReloadConfigOutputDoc(
            status="Config reloaded successfully",
            new_config=_config_data
        )
    except Exception as e:
        logging.error("[Router] reload-config error: %s", e)
        # Return an indicative doc or raise
        raise
