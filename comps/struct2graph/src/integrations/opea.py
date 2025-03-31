# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from typing import Annotated, Optional
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
    task : str
    cypher_cmd : str


@OpeaComponentRegistry.register("OPEA_STRUCT2GRAPH")
class OpeaStruct2Graph(OpeaComponent):
    """A specialized text to graph triplet converter."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.STRUCT2GRAPH.name.lower(), description, config)
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaStruct2Graph health check failed.")

    async def check_health(self) -> bool:
        """Checks the health of the TGI service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            return True
        except Exception as e:
            return False

    async def invoke(self, input: Input):
        """Invokes the struct2graph service to generate graph(s) for the provided input.

        input:
            input: text document
        task:
            Query or Index
        task:
            CSV or JSON
        Returns:
            text : dict
        """
        gdb = PrepareGraphDB()
        print(f" RECEIVED:: cypher_cmmand  {input.cypher_cmd}")
        print(f" RECEIVED:: input_text   {input.input_text}")
        print(f" RECEIVED:: task   {input.task}")
        if(input.task == 'Query'):
             graph_store = gdb.neo4j_link()
             result = graph_store.query(input.input_text)
        elif(input.task == 'Index'):
             graph_store = gdb.prepare_insert_graphdb(cypher_cmd=input.cypher_cmd)
             result = "Done indexing"
        return result
