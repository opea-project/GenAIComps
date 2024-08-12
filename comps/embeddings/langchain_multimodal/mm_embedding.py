# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from MMEmbeddings import BridgeTowerEmbeddings

from typing import Union, List
from langsmith import traceable
from comps import (
    EmbedDoc,
    ServiceType,
    TextDoc,
    ImageDoc,
    TextImageDoc,
    opea_microservices,
    register_microservice,
    statistics_dict,
)

MMDoc = Union[TextDoc, ImageDoc, TextImageDoc]

@register_microservice(
    name="opea_service@multimodal_embedding",
    service_type=ServiceType.EMBEDDING,
    endpoint="/v1/embeddings",
    host="0.0.0.0",
    port=6000,
    input_datatype=MMDoc,
    output_datatype=EmbedDoc,
)

@traceable(run_type="embedding")

def embedding(input: MMDoc) -> EmbedDoc:
    
    embeddings = BridgeTowerEmbeddings()

    if isinstance(input, TextDoc):
        # Handle text input
        embed_vector = embeddings.embed_query(input.text)
        res = EmbedDoc(text=input.text, embedding=embed_vector)
    
    #elif isinstance(input, TextImageDoc):
        # Handle text + image input
    #    embed_vector = embeddings.embed_image_text_pairs(input.texts, input.images, batch_size=2)
    #    res = EmbedDoc(text=input.texts, embedding=embed_vector)
    else:
        raise ValueError("Invalid input type")

    return res


if __name__ == "__main__":
   opea_microservices["opea_service@multimodal_embedding"].start()
