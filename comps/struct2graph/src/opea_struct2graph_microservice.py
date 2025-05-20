# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import sys

from fastapi.exceptions import HTTPException

from comps import CustomLogger, OpeaComponentLoader, opea_microservices, register_microservice
from comps.struct2graph.src.integrations.opea import Input, OpeaStruct2Graph

cur_path = pathlib.Path(__file__).parent.resolve()
comps_path = os.path.join(cur_path, "../../../")
sys.path.append(comps_path)

logger = CustomLogger("struct2graph")
logflag = os.getenv("LOGFLAG", False)

struct2graph_component_name = os.getenv("STRUCT2GRAPH_COMPONENT_NAME", "OPEA_STRUCT2GRAPH")

# Initialize OpeaComponentLoader
loader = OpeaComponentLoader(
    struct2graph_component_name,
    description=f"OPEA struct2graph Component: {struct2graph_component_name}",
)


@register_microservice(
    name="opea_service@struct2graph",
    endpoint="/v1/struct2graph",
    host="0.0.0.0",
    port=int(os.getenv("STRUCT2GRAPH_PORT", "8090")),
)
async def execute_agent(input: Input):
    """Execute triplet extraction from text file.
    This function takes an Input object containing the input text and database connection information.
    It uses the execute function from the struct2graph module to execute the graph query and returns the result.

    Args:
        input (Input): An Input object with the input text
        task (Input): type of task to perform index or query

    Returns:
        dict: A dictionary with head, tail and type linking head and tail
    """
    results = await loader.invoke(input)
    logger.info(f"PASSING BACK {results}")
    return {"result": results}


if __name__ == "__main__":
    logger.info("OPEA Struct2Graph Microservice is starting...")
    opea_microservices["opea_service@struct2graph"].start()
