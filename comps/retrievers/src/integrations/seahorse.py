# Copyright (C) 2026 Dnotitia
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

import requests
from fastapi import HTTPException
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from seahorse_vector_store import SeahorseVectorStore, SearchMode

from comps import CustomLogger, EmbedDoc, OpeaComponent, OpeaComponentRegistry, ServiceType

from .config import (
    EMBED_MODEL,
    HF_TOKEN,
    SEAHORSE_API_KEY,
    SEAHORSE_BASE_URL,
    SEAHORSE_EMBEDDING_MODE,
    SEAHORSE_SEARCH_MODE,
    TEI_EMBEDDING_ENDPOINT,
)

logger = CustomLogger("seahorse_retrievers")
logflag = os.getenv("LOGFLAG", False)
TEI_INFO_TIMEOUT_SECONDS = 10

SEARCH_MODE_MAP = {
    "dense": SearchMode.DENSE,
    "sparse": SearchMode.SPARSE,
    "hybrid": SearchMode.HYBRID,
}
SUPPORTED_SEARCH_TYPES = {
    "similarity",
    "similarity_score_threshold",
    "similarity_distance_threshold",
    "mmr",
}


@OpeaComponentRegistry.register("OPEA_RETRIEVER_SEAHORSE")
class OpeaSeahorseRetriever(OpeaComponent):
    """Seahorse Cloud managed vector search retriever.

    Connects to Seahorse Cloud API Gateway via langchain-seahorse SDK.
    SaaS — no self-hosted infrastructure required.

    Embedding mode (controlled by SEAHORSE_EMBEDDING_MODE env var):
    - "builtin": Uses Seahorse server-side embeddings for both indexing and search.
                 Calls similarity_search(query=text) so the server embeds the query.
    - "external": Initializes the SDK with a TEI or local HuggingFace embedder, but query-time
                  retrieval uses similarity_search_by_vector(embedding=vec) with the pre-computed
                  embedding passed in from the caller. External mode is always dense-only.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RETRIEVER.name.lower(), description, config)
        self.use_builtin = SEAHORSE_EMBEDDING_MODE == "builtin"
        self.embedder = self._initialize_embedder()
        self.search_mode = self._initialize_search_mode()
        self.vectorstore = self._initialize_vectorstore()
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaSeahorseRetriever health check failed.")

    @staticmethod
    def _require_env_config() -> None:
        missing = []
        if not SEAHORSE_BASE_URL:
            missing.append("SEAHORSE_BASE_URL")
        if not SEAHORSE_API_KEY:
            missing.append("SEAHORSE_API_KEY")
        if missing:
            raise RuntimeError(f"Missing required Seahorse configuration: {', '.join(missing)}")

    @staticmethod
    def _fetch_tei_model_id() -> str:
        try:
            response = requests.get(f"{TEI_EMBEDDING_ENDPOINT}/info", timeout=TEI_INFO_TIMEOUT_SECONDS)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} is not available: {exc}",
            ) from exc

        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} is not available.",
            )

        try:
            payload = response.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} returned invalid JSON.",
            ) from exc

        model_id = payload.get("model_id") if isinstance(payload, dict) else None
        if not model_id:
            raise HTTPException(
                status_code=400,
                detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} did not return model_id.",
            )
        return model_id

    @staticmethod
    def _normalize_search_type(search_type: str) -> str:
        if search_type not in SUPPORTED_SEARCH_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported search_type: {search_type}")
        if search_type == "mmr":
            logger.warning(
                "[ invoke ] MMR is not supported by Seahorse Cloud; falling back to similarity search. "
                "Diversity-aware retrieval will not be applied."
            )
            return "similarity"
        return search_type

    @staticmethod
    def _validate_external_embedding(embedding) -> None:
        if embedding is None or (isinstance(embedding, (list, tuple)) and len(embedding) == 0):
            raise HTTPException(
                status_code=400,
                detail="embedding must be provided when SEAHORSE_EMBEDDING_MODE=external.",
            )

    def _initialize_embedder(self):
        if self.use_builtin:
            if logflag:
                logger.info("[ init embedder ] Using Seahorse built-in embeddings (SEAHORSE_EMBEDDING_MODE=builtin)")
            return None

        if TEI_EMBEDDING_ENDPOINT:
            if logflag:
                logger.info(f"[ init embedder ] TEI_EMBEDDING_ENDPOINT: {TEI_EMBEDDING_ENDPOINT}")
            if not HF_TOKEN:
                raise HTTPException(
                    status_code=400,
                    detail="You MUST offer the `HF_TOKEN` when using `TEI_EMBEDDING_ENDPOINT`.",
                )
            model_id = self._fetch_tei_model_id()
            return HuggingFaceInferenceAPIEmbeddings(
                api_key=HF_TOKEN, model_name=model_id, api_url=TEI_EMBEDDING_ENDPOINT
            )

        if logflag:
            logger.info(f"[ init embedder ] LOCAL EMBED_MODEL: {EMBED_MODEL}")

        return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    def _initialize_search_mode(self) -> SearchMode:
        if SEAHORSE_SEARCH_MODE not in SEARCH_MODE_MAP:
            raise RuntimeError(
                f"Unknown SEAHORSE_SEARCH_MODE={SEAHORSE_SEARCH_MODE!r}. "
                f"Supported values: {sorted(SEARCH_MODE_MAP)}"
            )
        requested_mode = SEARCH_MODE_MAP[SEAHORSE_SEARCH_MODE]
        if not self.use_builtin and requested_mode != SearchMode.DENSE:
            logger.warning(
                "[ init ] external embedding mode only supports dense search. "
                f"Overriding SEAHORSE_SEARCH_MODE={SEAHORSE_SEARCH_MODE} to dense."
            )
            return SearchMode.DENSE
        return requested_mode

    def _initialize_vectorstore(self) -> SeahorseVectorStore:
        self._require_env_config()
        if logflag:
            logger.info(f"[ init ] SEAHORSE_BASE_URL: {SEAHORSE_BASE_URL}")
            logger.info(f"[ init ] SEAHORSE_EMBEDDING_MODE: {SEAHORSE_EMBEDDING_MODE}")
        kwargs = {
            "api_key": SEAHORSE_API_KEY,
            "base_url": SEAHORSE_BASE_URL,
        }
        if self.use_builtin:
            kwargs["use_builtin_embedding"] = True
        else:
            kwargs["embedding"] = self.embedder
            kwargs["use_builtin_embedding"] = False
        return SeahorseVectorStore(**kwargs)

    def check_health(self) -> bool:
        """Check Seahorse Cloud connectivity via the SDK's public health() API."""
        if logflag:
            logger.info("[ check health ] start to check health of Seahorse Cloud")
        try:
            self.vectorstore.health()
            if logflag:
                logger.info("[ check health ] Seahorse Cloud connected.")
            return True
        except Exception as e:
            logger.error(f"[ check health ] Failed to connect to Seahorse Cloud: {e}")
            return False

    @staticmethod
    def _passes_score_threshold(score: float, threshold: float, *, is_distance: bool) -> bool:
        """Match score semantics returned by the Seahorse SDK.

        Dense vector searches return distance values, where lower is better.
        Sparse and hybrid searches return similarity scores, where higher is better.
        """
        if is_distance:
            return score <= threshold
        return score >= threshold

    async def _search_builtin(self, input: EmbedDoc, search_type: str) -> list:
        """builtin mode: text query → server embeds with same built-in model → search."""
        if search_type == "similarity_score_threshold":
            docs_and_scores = await asyncio.to_thread(
                self.vectorstore.similarity_search_with_score,
                query=input.text,
                k=input.k,
                retrieval_mode=self.search_mode,
            )
            uses_distance = self.search_mode == SearchMode.DENSE
            return [
                doc
                for doc, score in docs_and_scores
                if self._passes_score_threshold(score, input.score_threshold, is_distance=uses_distance)
            ]

        if search_type == "similarity_distance_threshold":
            if input.distance_threshold is None:
                raise HTTPException(
                    status_code=400,
                    detail="distance_threshold must be provided for similarity_distance_threshold retriever",
                )
            if self.search_mode != SearchMode.DENSE:
                raise HTTPException(
                    status_code=400,
                    detail="similarity_distance_threshold is only supported for dense Seahorse search.",
                )
            docs_and_scores = await asyncio.to_thread(
                self.vectorstore.similarity_search_with_score,
                query=input.text,
                k=input.k,
                retrieval_mode=self.search_mode,
            )
            return [
                doc
                for doc, score in docs_and_scores
                if self._passes_score_threshold(score, input.distance_threshold, is_distance=True)
            ]

        return await asyncio.to_thread(
            self.vectorstore.similarity_search,
            query=input.text,
            k=input.k,
            retrieval_mode=self.search_mode,
        )

    async def _search_external(self, input: EmbedDoc, search_type: str) -> list:
        """external mode: use pre-computed OPEA TEI embedding vector for search.

        Note: similarity_search_by_vector() does not accept retrieval_mode parameter.
        External embedding mode is always dense-only, regardless of SEAHORSE_SEARCH_MODE.
        """
        self._validate_external_embedding(input.embedding)

        if search_type == "similarity_score_threshold":
            docs_and_scores = await asyncio.to_thread(
                self.vectorstore.similarity_search_by_vector_with_score,
                embedding=input.embedding,
                k=input.k,
            )
            return [
                doc
                for doc, score in docs_and_scores
                if self._passes_score_threshold(score, input.score_threshold, is_distance=True)
            ]

        if search_type == "similarity_distance_threshold":
            if input.distance_threshold is None:
                raise HTTPException(
                    status_code=400,
                    detail="distance_threshold must be provided for similarity_distance_threshold retriever",
                )
            docs_and_scores = await asyncio.to_thread(
                self.vectorstore.similarity_search_by_vector_with_score,
                embedding=input.embedding,
                k=input.k,
            )
            return [
                doc
                for doc, score in docs_and_scores
                if self._passes_score_threshold(score, input.distance_threshold, is_distance=True)
            ]

        return await asyncio.to_thread(
            self.vectorstore.similarity_search_by_vector,
            embedding=input.embedding,
            k=input.k,
        )

    async def invoke(self, input: EmbedDoc) -> list:
        """Search Seahorse Cloud for similar documents.

        Args:
            input (EmbedDoc): Query with embedding vector, search_type, k, etc.
        Returns:
            list: Retrieved documents.
        """
        if logflag:
            logger.info(input)

        search_type = self._normalize_search_type(input.search_type)

        if self.use_builtin:
            search_res = await self._search_builtin(input, search_type)
        else:
            search_res = await self._search_external(input, search_type)

        if logflag:
            logger.info(f"[ invoke ] retrieve result: {search_res}")

        return search_res
