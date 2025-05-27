# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from typing import Annotated, Optional

import requests
from pydantic import BaseModel, Field

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.struct2graph.src.integrations.graph_utils import PrepareGraphDB

global graph_store
logger = CustomLogger("comps-struct2graph")
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


class Input(BaseModel):
    input_text: str
    task: str
    cypher_cmd: str


@OpeaComponentRegistry.register("OPEA_STRUCT2GRAPH")
class OpeaStruct2Graph(OpeaComponent):
    """A specialized text to graph triplet converter."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.STRUCT2GRAPH.name.lower(), description, config)
        self.db = self.__initialize_db()
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaStruct2Graph health check failed.")

    def __initialize_db(self):
        """Initialize the graph database connection and return it."""
        logger.info("Initializing graph database...")
        return PrepareGraphDB()

    async def check_health(self) -> bool:
        """Checks the health of connection to the neo4j service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            logger.info("Performing health check...")
            neo4j_health_url = os.getenv("NEO4J_HEALTH_URL")
            response = requests.get(neo4j_health_url, timeout=5)
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Health check failed with status code: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    async def invoke(self, input: Input) -> dict:
        """Invokes the struct2graph service to generate graph(s) for the provided input.

        Args:
            input: Input object containing:
                - input_text: text document
                - task: Query or Index
                - cypher_cmd: CSV or JSON command

        Returns:
            dict: Result of the operation
        """
        logger.info("Starting struct2graph operation...")
        logger.debug(f"Received input: {input}")

        if input.task == "Query":
            logger.info("Executing query operation")
            graph_store = self.db.neo4j_link()
            result = graph_store.query(input.input_text)
            logger.info("Query executed successfully")

        elif input.task == "Index":
            logger.info("Executing index operation")
            graph_store = self.db.prepare_insert_graphdb(cypher_cmd=input.cypher_cmd)
            result = "Done indexing"
            logger.info("Indexing completed successfully")

        else:
            logger.error(f"Unsupported task type: {input.task}")
            raise ValueError(f"Unsupported task type: {input.task}")

        logger.info("Operation completed successfully")
        return result
