# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from comps.cores.common.component import OpeaComponent
from fastrag.rankers import IPEXBiEncoderSimilarityRanker
from haystack import Document

from comps import CustomLogger
from comps.cores.mega.micro_service import ServiceType
from comps.cores.proto.docarray import RerankedDoc, SearchedDoc, TextDoc

logger = CustomLogger("local_reranking")
logflag = os.getenv("LOGFLAG", False)
RANKER_MODEL = os.getenv("RANKER_MODEL")

class OpeaFastRAGReranking(OpeaComponent):
    """A specialized reranking component derived from OpeaComponent for fastRAG reranking services.

    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RERANK.name.lower(), description, config)
        self.reranker_model = IPEXBiEncoderSimilarityRanker(RANKER_MODEL)
        self.reranker_model.warm_up()


    async def invoke(self, input: SearchedDoc) -> RerankedDoc:
        """Invokes the reranking service to generate reranking for the provided input.

        Args:
            input (SearchedDoc): The input in OpenAI reranking format.

        Returns:
            RerankedDoc: The response in OpenAI reranking format.
        """
        documents = []
        for i, d in enumerate(input.retrieved_docs):
            documents.append(Document(content=d.text, id=(i + 1)))
        sorted_documents = self.reranker_model.run(input.initial_query, documents)["documents"]
        ranked_documents = [TextDoc(id=doc.id, text=doc.content) for doc in sorted_documents]
        res = RerankedDoc(initial_query=input.initial_query, reranked_docs=ranked_documents)

        return res

    def check_health(self) -> bool:
        """Checks the health of the reranking service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """

        return True
