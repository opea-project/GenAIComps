# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from MMEmbeddings import BridgeTowerEmbeddings

from typing import Union
from langsmith import traceable
from comps import (
    EmbedDoc1024,
    ServiceType,
    TextDoc,
    ImageDoc,
    TextImageDoc,
    opea_microservices,
    opea_telemetry,
    register_microservice,
    register_statistics,
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
    output_datatype=EmbedDoc1024,
)
@opea_telemetry
@traceable(run_type="embedding")
@register_statistics(names=["opea_service@multimodal_embedding"])

def embedding(input: MMDoc) -> EmbedDoc1024:
    start = time.time()

    if isinstance(input, TextDoc):
        # Handle text input
        embed_vector = BridgeTowerEmbeddings.embed_query(input.text)
        res = EmbedDoc1024(text=input.text, embedding=embed_vector)

    # function embed_image to be added 
    #elif isinstance(input, ImageDoc):
        # Handle image input
    #    embed_vector = BridgeTowerEmbeddings.embed_image(input.image_path)  
    #    res = EmbedDoc1024(text=input.image_path, embedding=embed_vector) 

    elif isinstance(input, TextImageDoc):
        # Handle text + image input
        embed_vector = BridgeTowerEmbeddings.embed_image_text_pairs(input.doc)  
        res = EmbedDoc1024(text=input.doc, embedding=embed_vector)
    else:
        raise ValueError("Invalid input type")


    statistics_dict["opea_service@embedding_multimodal"].append_latency(time.time() - start, None)
    return res


if __name__ == "__main__":
    embeddings = BridgeTowerEmbeddings(model_name="BridgeTower/bridgetower-large-itm-mlm-itc")
    opea_microservices["opea_service@multimodal_embedding"].start()
