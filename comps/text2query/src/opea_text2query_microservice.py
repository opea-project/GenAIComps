# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
OPEA Text2Query Microservice

This microservice provides text-to-query conversion capabilities supporting multiple query languages:
- SQL (Structured Query Language)
- Cypher (Neo4j graph database query language)
- Graph queries

The service dynamically loads the appropriate component based on the TEXT2QUERY_COMPONENT_NAME
environment variable and exposes a unfied REST API endpoint for query conversion.
"""

import os

from comps import CustomLogger, OpeaComponentLoader, opea_microservices, register_microservice
from comps.cores.proto.api_protocol import Text2QueryRequest

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

elif component_name == "OPEA_TEXT2QUERY_GRAPH":
    from comps.text2query.src.integrations.text2graph import OpeaText2GRAPH

else:
    raise ValueError(f"Unsupported TEXT2QUERY_COMPONENT_NAME: {component_name}")

# Initialize the OPEA component loader with the selected component
loader = OpeaComponentLoader(
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
    """
    Main service endpoint for text-to-query conversion.
    
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

if __name__ == "__main__":
    logger.info("OPEA Text2Query Microservice is starting...")
    opea_microservices["opea_service@text2query"].start()
