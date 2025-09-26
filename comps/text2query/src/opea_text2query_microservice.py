# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""OPEA Text2Query Microservice.

This microservice provides text-to-query conversion capabilities supporting multiple query languages:
- SQL (Structured Query Language)
- Cypher (Neo4j graph database query language)
- Graph queries

The service dynamically loads the appropriate component based on the TEXT2QUERY_COMPONENT_NAME
environment variable and exposes a unfied REST API endpoint for query conversion.
"""

import os

from fastapi import status
from fastapi.exceptions import HTTPException

from comps import CustomLogger, opea_microservices, register_microservice
from comps.cores.proto.api_protocol import Text2QueryRequest
from comps.text2query.src.opea_text2query_loader import OpeaText2QueryLoader

logger = CustomLogger("text2query")
logflag = os.getenv("LOGFLAG", False)

# Determine which text2query component to load based on environment configuration
# Default to SQL component if not specified
component_name = os.getenv("TEXT2QUERY_COMPONENT_NAME", "OPEA_TEXT2QUERY_SQL")

# Dynamically import the appropriate component implementation
if component_name == "OPEA_TEXT2QUERY_SQL":
    from comps.text2query.src.integrations.text2sql import OpeaText2SQL

elif component_name == "OPEA_TEXT2QUERY_CYPHER":
    from comps.text2query.src.integrations.text2cypher import OpeaText2Cypher

else:
    raise ValueError(f"Unsupported TEXT2QUERY_COMPONENT_NAME: {component_name}")

# Initialize the OPEA component loader with the selected component
loader = OpeaText2QueryLoader(
    component_name,
    description=f"OPEA TEXT2QUERY Component: {component_name}",
)


@register_microservice(
    name="opea_service@text2query",
    endpoint="/v1/text2query",
    host="0.0.0.0",
    port=9097,
)
async def execute_agent(request: Text2QueryRequest):
    """Main service endpoint for text-to-query conversion.

    This asynchronous function receives text2query requests and delegates
    processing to the loaded component implementation. The component handles
    the actual conversion from natural language text to the appropriate
    query language (SQL, Cypher, or Graph).

    Args:
        request (Text2QueryRequest): The incoming request containing text to convert

    Returns:
        The query conversion result from the loaded component
    """
    return await loader.invoke(request)


@register_microservice(
    name="opea_service@text2query",
    endpoint="/v1/db/health",
    host="0.0.0.0",
    port=9097,
)
async def db_connection_check(request: Text2QueryRequest):
    """Check the connection to the database.

    This function takes an Input object containing the database connection information.
    It uses the test_connection method of the PostgresConnection class to check if the connection is successful.

    Args:
        request (Text2QueryRequest): An Input object with the database connection information.

    Returns:
        dict: A dictionary with a 'status' key indicating whether the connection was successful or failed.
    """
    logger.info(f"Received input for connection check: {request}")
    if not isinstance(request, Text2QueryRequest):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Input type mismatch: expected Text2QueryRequest"
        )
    return await loader.db_connection_check(request)


if __name__ == "__main__":
    logger.info("OPEA Text2Query Microservice is starting...")
    opea_microservices["opea_service@text2query"].start()
