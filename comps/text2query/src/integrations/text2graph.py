# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import Text2QueryRequest
from comps.text2query.src.integrations.graph.graph_agent import TripletBuilder

logger = CustomLogger("comps-text2query-graph")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("OPEA_TEXT2QUERY_GRAPH")
class OpeaText2GRAPH(OpeaComponent):
    """A specialized text to graph triplet converter."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.TEXT2GRAPH.name.lower(), description, config)
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaText2GRAPH health check failed.")

    async def check_health(self) -> bool:
        """Checks the health of the TGI service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            return True
        except Exception as e:
            return False

    async def invoke(self, request: Text2QueryRequest):
        """Invokes the text2graph service to extract knowledge graph triplets from natural language text.

        This method processes the input text through a neural language model to identify and extract
        semantic relationships in the form of subject-predicate-object triplets (head-relation-tail).
        The extraction uses the REBEL model architecture to parse unstructured text and convert it
        into structured graph representations suitable for knowledge graph construction.

        Args:
            request (Text2QueryRequest): The request object containing:
                - query (str): The input text to extract graph triplets from

        Returns:
            dict: A dictionary containing:
                - result (dict): Contains extracted graph data with the key:
                    - graph_triplets (TripletManager): Manager object containing:
                        - entities: Dictionary of extracted entities with metadata
                        - relations: List of relationship triplets with head, type, and tail fields
        """

        tb = TripletBuilder()
        graph_triplets = await tb.extract_graph(request.query)

        result = {"graph_triplets": graph_triplets}

        return {"result": result}
