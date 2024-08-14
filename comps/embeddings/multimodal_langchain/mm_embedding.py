# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from multimodal_embeddings import BridgeTowerEmbedding

from typing import Union, List
from langsmith import traceable
from comps import (
    EmbedDoc,
    EmbedMultimodalDoc,
    ServiceType,
    TextDoc,
    ImageDoc,
    TextImageDoc,
    MultimodalDoc,
    opea_microservices,
    register_microservice,
    statistics_dict,
)



@register_microservice(
    name="opea_service@multimodal_embedding",
    service_type=ServiceType.EMBEDDING,
    endpoint="/v1/embeddings",
    host="0.0.0.0",
    port=6000,
    input_datatype=MultimodalDoc,
    output_datatype=EmbedMultimodalDoc,
)

@traceable(run_type="embedding")

def embedding(input: MultimodalDoc) -> EmbedDoc:
    start = time.time()
    # print("HELLLLLOOOOOOOO")
    # print(input)
    # print(type(input))

    embeddings = BridgeTowerEmbedding()
    
    if isinstance(input, TextDoc):
        # Handle text input
        embed_vector = embeddings.embed_query(input.text)
        res = EmbedDoc(text=input.text, embedding=embed_vector)

    elif isinstance(input, TextImageDoc):
        # Handle text + image input
        embed_vector = embeddings.embed_image_text_pairs(input.texts, input.images, batch_size=2)
        res = EmbedMultimodalDoc(texts=input.texts, image_paths=input.images, embedding=embed_vector)
    else:
        raise ValueError("Invalid input type")

    return res

if __name__ == "__main__":
    opea_microservices["opea_service@multimodal_embedding"].start()
