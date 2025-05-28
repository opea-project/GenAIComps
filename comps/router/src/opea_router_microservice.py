# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import yaml

from comps import (
    CustomLogger,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
)
from comps.cores.proto.api_protocol import RouteEndpointDoc
from comps.router.src.integrations.controllers.controller_factory import ControllerFactory

# Set up logging
logger = CustomLogger("opea_router_microservice")
logflag = os.getenv("LOGFLAG", False)

CONFIG_PATH = os.getenv("CONFIG_PATH")

_config_data = {}
_controller_factory = None
_controller = None


def _load_config():
    global _config_data, _controller_factory, _controller

    try:
        with open(CONFIG_PATH, "r") as f:
            new_data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise RuntimeError(f"Failed to load config: {e}")

    _config_data = new_data
    logger.info(f"[Router] Loaded config data from: {CONFIG_PATH}")

    if _controller_factory is None:
        _controller_factory = ControllerFactory()

    model_map = _config_data.get("model_map", {})
    controller_type = os.getenv("CONTROLLER_TYPE") or _config_data.get("controller_type", "routellm")

    # look up the correct controller-config path
    try:
        controller_config_path = _config_data["controller_config_paths"][controller_type]
    except KeyError:
        raise RuntimeError(f"No config path for controller_type='{controller_type}' in global config")

    _controller = _controller_factory.factory(controller_config=controller_config_path, model_map=model_map)

    logger.info("[Router] Controller re-initialized successfully.")


# Initial config load at startup
_load_config()


@register_microservice(
    name="opea_service@router",
    service_type=ServiceType.LLM,
    endpoint="/v1/route",
    host="0.0.0.0",
    port=6000,
    input_datatype=TextDoc,
    output_datatype=RouteEndpointDoc,
)
def route_microservice(input: TextDoc) -> RouteEndpointDoc:
    """Microservice that decides which model endpoint is best for the given text input.

    Returns only the route URL (does not forward).
    """
    if not _controller:
        raise RuntimeError("Controller is not initialized — config load failed?")

    query_content = input.text
    messages = [{"content": query_content}]

    try:
        endpoint = _controller.route(messages)
        if not endpoint:
            raise ValueError("No suitable model endpoint found.")
        return RouteEndpointDoc(url=endpoint)

    except Exception as e:
        logger.error(f"[Router] Error during model routing: {e}")
        raise


if __name__ == "__main__":
    logger.info("OPEA Router Microservice is starting...")
    opea_microservices["opea_service@router"].start()
