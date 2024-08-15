# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from embeddings_clip import vCLIP
from typing import Union
from langsmith import traceable
from comps import (
    EmbedDoc,
    ServiceType,
    TextDoc,
    MultimodalTextInput,
    opea_microservices,
    opea_telemetry,
    register_microservice,
    register_statistics,
    statistics_dict,
)

@register_microservice(
    name="opea_service@embedding_multimodal",
    service_type=ServiceType.EMBEDDING,
    endpoint="/v1/embeddings",
    host="0.0.0.0",
    port=6000,
    input_datatype=MultimodalTextInput,
    output_datatype=EmbedDoc,
)
@opea_telemetry
@traceable(run_type="embedding")
@register_statistics(names=["opea_service@embedding_multimodal"])

def embedding(input: MultimodalTextInput) -> EmbedDoc:
    start = time.time()
   
    if isinstance(input, MultimodalTextInput):
        # Handle text input
        embed_vector = embeddings.get_text_embeddings(input.text).tolist()[0]
        res = EmbedDoc(text=input.text, embedding=embed_vector, constraints={})

    else:
        raise ValueError("Invalid input type")
        

    statistics_dict["opea_service@embedding_multimodal"].append_latency(time.time() - start, None)
    return res



if __name__ == "__main__":
    embeddings = vCLIP({"model_name": "openai/clip-vit-base-patch32", "num_frm": 4})
    opea_microservices["opea_service@embedding_multimodal"].start()
    
