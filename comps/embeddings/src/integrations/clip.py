# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Union

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoProcessor, AutoTokenizer, CLIPModel

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import EmbeddingRequest, EmbeddingResponse, EmbeddingResponseData

logger = CustomLogger("opea_multimodal_embedding_clip")
logflag = os.getenv("LOGFLAG", False)


model_name = "openai/clip-vit-base-patch32"

clip = CLIPModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


class vCLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_frm = cfg["num_frm"]
        self.model_name = cfg["model_name"]

    def embed_query(self, texts):
        """Input is list of texts."""
        text_inputs = tokenizer(texts, padding=True, return_tensors="pt")
        text_features = clip.get_text_features(**text_inputs)
        return text_features

    def get_embedding_length(self):
        text_features = self.embed_query("sample_text")
        return text_features.shape[1]

    def get_image_embeddings(self, images):
        """Input is list of images."""
        image_inputs = processor(images=images, return_tensors="pt")
        image_features = clip.get_image_features(**image_inputs)
        return image_features

    def get_video_embeddings(self, frames_batch):
        """Input is list of list of frames in video."""
        self.batch_size = len(frames_batch)
        vid_embs = []
        for frames in frames_batch:
            frame_embeddings = self.get_image_embeddings(frames)
            frame_embeddings = rearrange(frame_embeddings, "(b n) d -> b n d", b=len(frames_batch))
            # Normalize, mean aggregate and return normalized video_embeddings
            frame_embeddings = frame_embeddings / frame_embeddings.norm(dim=-1, keepdim=True)
            video_embeddings = frame_embeddings.mean(dim=1)
            video_embeddings = video_embeddings / video_embeddings.norm(dim=-1, keepdim=True)
            vid_embs.append(video_embeddings)
        return torch.cat(vid_embs, dim=0)


@OpeaComponentRegistry.register("OPEA_CLIP_EMBEDDING")
class OpeaClipEmbedding(OpeaComponent):
    """A specialized embedding component derived from OpeaComponent for CLIP embedding services.

    This class initializes and configures the CLIP embedding service using the vCLIP model.
    It also performs a health check during initialization and logs an error if the check fails.

    Attributes:
        embeddings (vCLIP): An instance of the vCLIP model used for generating embeddings.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.EMBEDDING.name.lower(), description, config)
        self.embeddings = vCLIP({"model_name": "openai/clip-vit-base-patch32", "num_frm": 4})

        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaClipEmbedding health check failed.")

    async def invoke(self, input: EmbeddingRequest) -> EmbeddingResponse:
        """Invokes the embedding service to generate embeddings for the provided input.

        Args:
            input (EmbeddingRequest): The input in OpenAI embedding format, including text(s) and optional parameters like model.

        Returns:
            EmbeddingResponse: The response in OpenAI embedding format, including embeddings, model, and usage information.
        """
        # Parse input according to the EmbeddingRequest format
        if isinstance(input.input, str):
            texts = [input.input.replace("\n", " ")]
        elif isinstance(input.input, list):
            if all(isinstance(item, str) for item in input.input):
                texts = [text.replace("\n", " ") for text in input.input]
            else:
                raise ValueError("Invalid input format: Only string or list of strings are supported.")
        else:
            raise TypeError("Unsupported input type: input must be a string or list of strings.")
        embed_vector = self.get_embeddings(texts)
        if input.dimensions is not None:
            embed_vector = [embed_vector[i][: input.dimensions] for i in range(len(embed_vector))]

        # for standard openai embedding format
        res = EmbeddingResponse(
            data=[EmbeddingResponseData(index=i, embedding=embed_vector[i]) for i in range(len(embed_vector))]
        )
        return res

    def check_health(self) -> bool:
        """Checks if the embedding model is healthy.

        Returns:
            bool: True if the embedding model is initialized, False otherwise.
        """
        if self.embeddings:
            return True
        else:
            return False

    def get_embeddings(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Generates embeddings for input text.

        Args:
            text (Union[str, List[str]]): Input text or list of texts.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        texts = [text] if isinstance(text, str) else text
        embed_vector = self.embeddings.embed_query(texts).tolist()
        return embed_vector
