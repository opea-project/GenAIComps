# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from embeddings_clip import tCLIP
from typing import Union
from einops import rearrange
from langsmith import traceable
from comps import (
    EmbedDoc,
    ServiceType,
    TextDoc,
    ImageData,
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
    input_datatype=Union[TextDoc,ImageData],
    output_datatype=EmbedDoc,
)
@opea_telemetry
@traceable(run_type="embedding")
@register_statistics(names=["opea_service@embedding_multimodal"])

def embedding(input: Union[TextDoc,ImageData]) -> EmbedDoc:
    start = time.time()
   
    if isinstance(input, TextDoc):
        # Handle text input
        embed_vector = embeddings.get_text_embeddings(input.text).tolist()[0]
        print('done clip')
        res = EmbedDoc(text=input.text, embedding=embed_vector)
    
    elif isinstance(input, ImageData):
        # Handle text input
        batch_size = len(input.image)
        vid_embs = []
        for frames in input.image:
            frame_embeddings = embeddings.get_image_embeddings(frames)
            frame_embeddings = rearrange(frame_embeddings, "(b n) d -> b n d", b=batch_size)
            frame_embeddings = frame_embeddings / frame_embeddings.norm(dim=-1, keepdim=True) 
            video_embeddings = frame_embeddings.mean(dim=1)
            video_embeddings = video_embeddings / video_embeddings.norm(dim=-1, keepdim=True)
            vid_embs.append(video_embeddings)
        res = EmbedDoc(text='video embeddings', embedding=torch.cat(vid_embs, dim=0)
        
    else:
        raise ValueError("Invalid input type")
        

    statistics_dict["opea_service@embedding_multimodal"].append_latency(time.time() - start, None)
    return res



if __name__ == "__main__":
    embeddings = tCLIP({"model_name": "openai/clip-vit-base-patch32", "num_frm": 4})
    opea_microservices["opea_service@embedding_multimodal"].start()
    