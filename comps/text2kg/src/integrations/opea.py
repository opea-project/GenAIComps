# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from typing import Annotated, Optional

import requests
from langchain.agents.agent_types import AgentType
from langchain_huggingface import HuggingFaceEndpoint
from pydantic import BaseModel, Field

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.text2kg.src.integrations.kg_graph_agent import GenerateKG

logger = CustomLogger("comps-text2kg")
logflag = os.getenv("LOGFLAG", False)

graph_params = {
    "max_string_length": 3600,
}

generation_params = {
    "max_new_tokens": 1024,
    "top_k": 10,
    "top_p": 0.95,
    "temperature": 0.01,
    "repetition_penalty": 1.03,
    "streaming": True,
}

TGI_LLM_ENDPOINT = os.environ.get("TGI_LLM_ENDPOINT")

llm = HuggingFaceEndpoint(
    endpoint_url=TGI_LLM_ENDPOINT,
    task="text-generation",
    **generation_params,
)


class Input(BaseModel):
    input_text: str


@OpeaComponentRegistry.register("OPEA_TEXT2KG")
class OpeaText2KG(OpeaComponent):
    """A specialized text to graph triplet converter."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.TEXT2KG.name.lower(), description, config)
        global neo4j_index
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaText2KG health check failed.")
        gdb = GenerateKG(
            data_directory="data/", embedding_model="BAAI/bge-small-en-v1.5", llm="HuggingFaceH4/zephyr-7b-alpha"
        )
        neo4j_index = gdb.prepare_and_save_graphdb()

    async def check_health(self) -> bool:
        """Checks the health of connection to the neo4j service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            neo4j_health_url = os.getenv("NEO4J_HEALTH_URL")
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

        query_engine = neo4j_index.as_query_engine(include_text=False, response_mode="tree_summarize")

        result = query_engine.query(input_text)
        print(result)

        return result
