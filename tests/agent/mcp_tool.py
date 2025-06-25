# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os

from comps import CustomLogger, ServiceType, opea_microservices, register_microservice
from comps.cores.mega.constants import MCPFuncType

logger = CustomLogger("comps-react-agent")
logflag = os.getenv("LOGFLAG", False)


@register_microservice(
    name="opea_service@mcp_rag_tool",
    service_type=ServiceType.UNDEFINED,
    endpoint="/v1/mcp_rag_tool/search_kb",
    host="0.0.0.0",
    port=8000,
    input_datatype=str,
    output_datatype=str,
    enable_mcp=True,
    mcp_func_type=MCPFuncType.TOOL,
    description="Returns the information about OPEA",
)
def search_kb(query: str) -> str:
    """Search the knowledge base for a given query."""
    ret_text = """
    The Linux Foundation AI & Data announced the Open Platform for Enterprise AI (OPEA) as its latest Sandbox Project.
    OPEA aims to accelerate secure, cost-effective generative AI (GenAI) deployments for businesses by driving interoperability across a diverse and heterogeneous ecosystem, starting with retrieval-augmented generation (RAG).
    """
    return ret_text


if __name__ == "__main__":
    logger.info("OPEA math_tools Microservice is starting....")
    opea_microservices["opea_service@mcp_rag_tool"].start()
