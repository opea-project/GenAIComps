# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import sys

from fastapi.exceptions import HTTPException

from comps import CustomLogger, OpeaComponentLoader, opea_microservices, register_microservice
from comps.text2kg.src.integrations.opea import Input, OpeaText2KG

cur_path = pathlib.Path(__file__).parent.resolve()
comps_path = os.path.join(cur_path, "../../../")
sys.path.append(comps_path)

logger = CustomLogger("text2kg")
logflag = os.getenv("LOGFLAG", False)

text2kg_component_name = os.getenv("TEXT2KG_COMPONENT_NAME", "OPEA_TEXT2KG")

# Initialize OpeaComponentLoader
loader = OpeaComponentLoader(
    text2kg_component_name,
    description=f"OPEA text2kg Component: {text2kg_component_name}",
)


@register_microservice(
    name="opea_service@text2kg",
    endpoint="/v1/text2kg",
    host="0.0.0.0",
    port=os.getenv("TEXT2KG_PORT"),
)
async def execute_agent(input_text: str):
    """Execute triplet extraction from text file.

    This function takes an Input object containing the input text and database connection information.
    It uses the execute function from the text2kg module to execute the graph query and returns the result.
    Args:
        input (Input): An Input object with the input text
    Returns:
        dict: A dictionary with head, tail and type linking head and tail
    """
    print("===============================================================")
    print("===================ENTERING THIS EXECUTE AGENT=================")
    print("===============================================================")
    result = await loader.invoke(input_text)
    return {"result": result}


if __name__ == "__main__":
    logger.info("OPEA Text2KG Microservice is starting...")
    opea_microservices["opea_service@text2kg"].start()
