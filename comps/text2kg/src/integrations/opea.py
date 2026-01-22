# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from typing import Annotated, Optional

import requests
from pydantic import BaseModel, Field

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType

logger = CustomLogger("comps-text2kg")
logflag = os.getenv("LOGFLAG", False)

graph_params = {
    "max_string_length": 3600,
}

TGI_LLM_ENDPOINT = os.environ.get("TGI_LLM_ENDPOINT")


class Input(BaseModel):
    input_text: str


# Global variable to store the index, initialized lazily
_neo4j_index = None
_gdb = None


def _get_neo4j_index():
    """Lazily initialize the knowledge graph index."""
    global _neo4j_index, _gdb
    if _neo4j_index is None:
        from comps.text2kg.src.integrations.kg_graph_agent import GenerateKG

        _gdb = GenerateKG(
            data_directory="data/", embedding_model="BAAI/bge-small-en-v1.5", llm_endpoint_url=TGI_LLM_ENDPOINT
        )
        _neo4j_index = _gdb.prepare_and_save_graphdb()
    return _neo4j_index


@OpeaComponentRegistry.register("OPEA_TEXT2KG")
class OpeaText2KG(OpeaComponent):
    """A specialized text to graph triplet converter."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.TEXT2KG.name.lower(), description, config)
        # Defer graph initialization to first invocation
        logger.info("OpeaText2KG initialized. Graph will be created on first query.")

    async def check_health(self) -> bool:
        """Checks the health of connection to the neo4j service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            neo4j_health_url = os.getenv("NEO4J_HEALTH_URL")
            if not neo4j_health_url:
                logger.warning("NEO4J_HEALTH_URL not set, skipping health check.")
                return True
            response = requests.get(neo4j_health_url, timeout=5)
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Health check failed with status code: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def invoke(self, input_text: str):
        """Invokes the text2kg service to generate graph(s) for the provided input.

        input:
            input: text document
        Returns:
            text : dict
        """
        # Lazy initialization of the knowledge graph
        neo4j_index = _get_neo4j_index()

        query_engine = neo4j_index.as_query_engine(include_text=False, response_mode="tree_summarize")

        result = query_engine.query(input_text)
        print(result)

        return result
