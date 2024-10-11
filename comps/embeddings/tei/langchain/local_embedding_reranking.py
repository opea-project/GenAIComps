# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import os
from enum import Enum
from pathlib import Path
from typing import Union

import torch
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
from sentence_transformers.models import Pooling
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from comps import (
    CustomLogger,
    EmbedDoc,
    LLMParamsDoc,
    RerankedDoc,
    SearchedDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    opea_telemetry,
    register_microservice,
)
from comps.cores.proto.api_protocol import (
    ChatCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
)

logger = CustomLogger("local_embedding_reranking")
logflag = os.getenv("LOGFLAG", False)

# keep it consistent for different routers for now
STATIC_BATCHING_TIMEOUT = 1
STATIC_BATCHING_MAX_BATCH_SIZE = 32


class EmbeddingModel:
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "cls",
        trust_remote: bool = False,
    ):
        if device == torch.device("hpu"):
            adapt_transformers_to_gaudi()
        model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote).to(dtype).to(device)
        self.model = model
        if device == torch.device("hpu"):
            logger.info("Use graph mode for HPU")
            model = wrap_in_hpu_graph(model, disable_tensor_cache=True)
        self.hidden_size = model.config.hidden_size
        self.pooling = Pooling(self.hidden_size, pooling_mode=pool)
        position_offset = 0
        model_type = model.config.model_type
        if model_type in ["xlm-roberta", "camembert", "roberta"]:
            position_offset = model.config.pad_token_id + 1
        max_input_length = 0
        if hasattr(model.config, "max_seq_length"):
            max_input_length = model.config.max_seq_length
        else:
            max_input_length = model.config.max_position_embeddings - position_offset
        self.max_input_length = max_input_length
        self.has_position_ids = inspect.signature(model.forward).parameters.get("position_ids", None) is not None
        self.has_token_type_ids = inspect.signature(model.forward).parameters.get("token_type_ids", None) is not None

    def embed(self, batch):
        output = self.model(**batch)
        pooling_features = {
            "token_embeddings": output[0],
            "attention_mask": batch.attention_mask,
        }
        embedding = self.pooling.forward(pooling_features)["sentence_embedding"]
        cpu_results = embedding.reshape(-1).tolist()

        return [cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size] for i in range(len(batch.input_ids))]


class RerankingModel:
    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        if device == torch.device("hpu"):
            adapt_transformers_to_gaudi()

        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = model.to(dtype).to(device)

        if device == torch.device("hpu"):
            logger.info("Use graph mode for HPU")
            model = wrap_in_hpu_graph(model, disable_tensor_cache=True)
        self.model = model
        self.hidden_size = model.config.hidden_size
        position_offset = 0
        model_type = model.config.model_type
        if model_type in ["xlm-roberta", "camembert", "roberta"]:
            position_offset = model.config.pad_token_id + 1
        max_input_length = 0
        if hasattr(model.config, "max_seq_length"):
            max_input_length = model.config.max_seq_length
        else:
            max_input_length = model.config.max_position_embeddings - position_offset
        self.max_input_length = max_input_length
        self.has_position_ids = inspect.signature(model.forward).parameters.get("position_ids", None) is not None
        self.has_token_type_ids = inspect.signature(model.forward).parameters.get("token_type_ids", None) is not None

    def embed(self, batch):
        raise NotImplementedError(f"Embed is not a valid operation for model type {self.model.config.model_type}")

    def predict(self, batch):
        output = self.model(**batch, return_dict=True)
        all_scores = output.logits.tolist()
        return all_scores


async def static_batching_infer(service_type: Enum, batch: list[dict]):
    if logflag:
        logger.info(f"{service_type} {len(batch)} request inference begin >>>")

    if service_type == ServiceType.EMBEDDING:
        # TODO PADDING TO (MAX_BS, MAX_LEN)
        sentences = [req["request"].text for req in batch]
        with torch.no_grad():
            encoded_input = embedding_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(
                device="hpu"
            )
            results = embedding_model.embed(encoded_input)

        return [EmbedDoc(text=txt, embedding=embed_vector) for txt, embed_vector in zip(sentences, results)]
    elif service_type == ServiceType.RERANK:
        pairs = []
        doc_lengths = []
        for req in batch:
            doc_len = len(req["request"].retrieved_docs)
            doc_lengths.append(doc_len)
            for idx in range(doc_len):
                pairs.append([req["request"].initial_query, req["request"].retrieved_docs[idx].text])

        with torch.no_grad():
            inputs = reranking_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(
                "hpu"
            )
            scores = reranking_model.predict(inputs)

        # reduce each query's best related doc
        final_results = []
        start = 0
        for idx, doc_len in enumerate(doc_lengths):
            req_scores = scores[start : start + doc_len]
            cur_req = batch[idx]["request"]
            docs: list[TextDoc] = cur_req.retrieved_docs[0:doc_len]
            docs = [doc.text for doc in docs]
            # sort and select top n docs
            top_n_docs = sorted(list(zip(docs, req_scores)), key=lambda x: x[1], reverse=True)[: cur_req.top_n]
            top_n_docs: list[str] = [tupl[0] for tupl in top_n_docs]
            final_results.append(LLMParamsDoc(query=cur_req.initial_query, documents=top_n_docs))

            start += doc_len

        return final_results


@register_microservice(
    name="opea_service@local_embedding_reranking",
    service_type=ServiceType.EMBEDDING,
    endpoint="/v1/embeddings",
    host="0.0.0.0",
    port=6001,
    static_batching=True,
    static_batching_timeout=STATIC_BATCHING_TIMEOUT,
    static_batching_max_batch_size=STATIC_BATCHING_MAX_BATCH_SIZE,
)
async def embedding(
    input: Union[TextDoc, EmbeddingRequest, ChatCompletionRequest]
) -> Union[EmbedDoc, EmbeddingResponse, ChatCompletionRequest]:

    if logflag:
        logger.info(input)
    # Create a future for this specific request
    response_future = asyncio.get_event_loop().create_future()

    cur_microservice = opea_microservices["opea_service@local_embedding_reranking"]
    cur_microservice.static_batching_infer = static_batching_infer
    async with cur_microservice.buffer_lock:
        cur_microservice.request_buffer[ServiceType.EMBEDDING].append({"request": input, "response": response_future})

    # Wait for batch inference to complete and return results
    result = await response_future

    return result


@register_microservice(
    name="opea_service@local_embedding_reranking",
    service_type=ServiceType.RERANK,
    endpoint="/v1/reranking",
    host="0.0.0.0",
    port=6001,
    input_datatype=SearchedDoc,
    output_datatype=LLMParamsDoc,
    static_batching=True,
    static_batching_timeout=STATIC_BATCHING_TIMEOUT,
    static_batching_max_batch_size=STATIC_BATCHING_MAX_BATCH_SIZE,
)
async def reranking(input: SearchedDoc) -> LLMParamsDoc:

    if logflag:
        logger.info(input)
    # Create a future for this specific request
    response_future = asyncio.get_event_loop().create_future()

    cur_microservice = opea_microservices["opea_service@local_embedding_reranking"]
    cur_microservice.static_batching_infer = static_batching_infer
    async with cur_microservice.buffer_lock:
        cur_microservice.request_buffer[ServiceType.RERANK].append({"request": input, "response": response_future})

    # Wait for batch inference to complete and return results
    result = await response_future

    return result


if __name__ == "__main__":
    embedding_model = EmbeddingModel(model_path="BAAI/bge-base-zh-v1.5", device="hpu", dtype=torch.float32)
    embedding_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
    # sentences = ["sample-1", "sample-2"]
    # encoded_input = embedding_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device="hpu")
    # results = embedding_model.embed(encoded_input)
    # print(results)
    reranking_model = RerankingModel(model_path="BAAI/bge-reranker-base", device="hpu", dtype=torch.float32)
    reranking_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")

    # pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
    # with torch.no_grad():
    #     inputs = reranking_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to("hpu")
    #     scores = reranking_model.predict(inputs)
    #     print(scores)
    opea_microservices["opea_service@local_embedding_reranking"].start()
