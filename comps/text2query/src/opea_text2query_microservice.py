# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from comps import CustomLogger, OpeaComponentLoader, opea_microservices, register_microservice
from comps.cores.proto.api_protocol import Text2QueryRequest

# cur_path = pathlib.Path(__file__).parent.resolve()
# comps_path = os.path.join(cur_path, "../../../")
# sys.path.append(comps_path)

logger = CustomLogger("text2query")
logflag = os.getenv("LOGFLAG", False)

text2query_component_name = os.getenv("TEXT2QUERY_COMPONENT_NAME", "OPEA_TEXT2QUERY_SQL")
if text2query_component_name == "OPEA_TEXT2QUERY_SQL":
    from comps.text2query.src.integrations.sql.text2sql import OpeaText2SQL
elif text2query_component_name == "OPEA_TEXT2QUERY_CYPHER":
    from comps.text2query.src.integrations.cypher.native import OpeaText2Cypher

# Initialize OpeaComponentLoader
loader = OpeaComponentLoader(
    text2query_component_name,
    description=f"OPEA TEXT2QUERY Component: {text2query_component_name}",
)


@register_microservice(
    name="opea_service@text2query",
    endpoint="/v1/text2query",
    host="0.0.0.0",
    port=9097,
)
async def execute_agent(request: Text2QueryRequest):
    return await loader.invoke(request)


if __name__ == "__main__":
    logger.info("OPEA Text2Query Microservice is starting...")
    opea_microservices["opea_service@text2query"].start()
