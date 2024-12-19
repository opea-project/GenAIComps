# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import time

from comps import (
    OpeaComponentController,
    CustomLogger,
    SDInputs,
    SDOutputs,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from integrations import OpeaText2image

logger = CustomLogger("opea_text2image_microservice")

# Initialize OpeaComponentController
controller = OpeaComponentController()


@register_microservice(
    name="opea_service@text2image",
    service_type=ServiceType.TEXT2IMAGE,
    endpoint="/v1/text2image",
    host="0.0.0.0",
    port=9379,
    input_datatype=SDInputs,
    output_datatype=SDOutputs,
)
@register_statistics(names=["opea_service@text2image"])
async def text2image(input: SDInputs):
    start = time.time()
    try:
        # Use the controller to invoke the active component
        results = await controller.invoke(input)
        statistics_dict["opea_service@text2image"].append_latency(time.time() - start, None)
        return results
    except Exception as e:
        logger.error(f"Error during text2image invocation: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--use_hpu_graphs", default=False, action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()

    # Register components
    try:
        # Instantiate Embedding components
        opea_text2image = OpeaText2image(
            name="OpeaText2image",
            description="OPEA Text2image Service",
            config=args.__dict__
        )

        # Register components with the controller
        controller.register(opea_text2image)

        # Discover and activate a healthy component
        controller.discover_and_activate()
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")

    logger.info("Text2image server started.")
    opea_microservices["opea_service@text2image"].start()
