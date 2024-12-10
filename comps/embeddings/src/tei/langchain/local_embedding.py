# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from langchain_huggingface import HuggingFaceEmbeddings

from comps import (
    CustomLogger,
    EmbedDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    opea_telemetry,
    register_microservice,
)

logger = CustomLogger("local_embedding")
logflag = os.getenv("LOGFLAG", False)


@register_microservice(
    name="opea_service@local_embedding",
    service_type=ServiceType.EMBEDDING,
    endpoint="/v1/embeddings",
    host="0.0.0.0",
    port=6000,
    input_datatype=TextDoc,
    output_datatype=EmbedDoc,
)
@opea_telemetry
async def embedding(input: TextDoc) -> EmbedDoc:
    if logflag:
        logger.info(input)
    embed_vector = await embeddings.aembed_query(input.text)
    res = EmbedDoc(text=input.text, embedding=embed_vector)
    if logflag:
        logger.info(res)
    return res


if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    opea_microservices["opea_service@local_embedding"].start()
