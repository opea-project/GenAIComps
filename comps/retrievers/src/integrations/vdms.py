# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
from typing import Any, Dict, List

import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from decord import VideoReader, cpu
from einops import rearrange
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_vdms.vectorstores import VDMS, VDMS_Client
from pydantic import BaseModel, model_validator
from torch import cat as torch_cat
from transformers import AutoProcessor, AutoTokenizer, CLIPModel

from comps import CustomLogger, EmbedDoc, OpeaComponent, OpeaComponentRegistry, ServiceType

from .config import (
    DISTANCE_STRATEGY,
    EMBED_MODEL,
    HUGGINGFACEHUB_API_TOKEN,
    SEARCH_ENGINE,
    TEI_EMBEDDING_ENDPOINT,
    VDMS_HOST,
    VDMS_INDEX_NAME,
    VDMS_PORT,
    VDMS_USE_CLIP,
)

logger = CustomLogger("vdms_retrievers")
logflag = os.getenv("LOGFLAG", False)
toPIL = T.ToPILImage()


@OpeaComponentRegistry.register("OPEA_RETRIEVER_VDMS")
class OpeaVDMsRetriever(OpeaComponent):
    """A specialized retriever component derived from OpeaComponent for vdms retriever services.

    Attributes:
        client (VDMS): An instance of the vdms client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RETRIEVER.name.lower(), description, config)

        self.embedder = self._initialize_embedder()
        self.client = VDMS_Client(host=VDMS_HOST, port=VDMS_PORT)
        self.vector_db = self._initialize_vector_db()
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaVDMsRetriever health check failed.")

    def _initialize_embedder(self):
        if VDMS_USE_CLIP:
            meanclip_cfg = {
                "model_name": "openai/clip-vit-base-patch32",
                "num_frm": 64,
            }
            video_retriever_model = vCLIP(meanclip_cfg)  # , device="cpu")
            embeddings = vCLIPEmbeddings(model=video_retriever_model)

        elif TEI_EMBEDDING_ENDPOINT:
            # create embeddings using TEI endpoint service
            if logflag:
                logger.info(f"[ init embedder ] TEI_EMBEDDING_ENDPOINT:{TEI_EMBEDDING_ENDPOINT}")
            from langchain_huggingface import HuggingFaceEndpointEmbeddings

            embeddings = HuggingFaceEndpointEmbeddings(model=TEI_EMBEDDING_ENDPOINT)
        else:
            # create embeddings using local embedding model
            if logflag:
                logger.info(f"[ init embedder ] LOCAL_EMBEDDING_MODEL:{EMBED_MODEL}")
            embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        return embeddings

    def _initialize_vector_db(self) -> VDMS:
        """Initializes the vdms client."""
        vector_db = VDMS(
            client=self.client,
            embedding=self.embedder,
            collection_name=VDMS_INDEX_NAME,
            distance_strategy=DISTANCE_STRATEGY,
            engine=SEARCH_ENGINE,
        )
        return vector_db

    def check_health(self) -> bool:
        """Checks the health of the retriever service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if logflag:
            logger.info("[ check health ] start to check health of vdms")
        try:
            if self.vector_db:
                logger.info("[ check health ] Successfully connected to VDMs!")
                return True
            else:
                logger.info("[ check health ] Failed to connect to VDMs.")
                return False
        except Exception as e:
            logger.info(f"[ check health ] Failed to connect to VDMs: {e}")
            return False

    async def invoke(self, input: EmbedDoc) -> list:
        """Search the VDMs index for the most similar documents to the input query.

        Args:
            input (EmbedDoc): The input query to search for.
        Output:
            list: The retrieved documents.
        """
        if logflag:
            logger.info(input)

        if input.search_type == "similarity":
            search_res = self.vector_db.similarity_search_by_vector(
                embedding=input.embedding, k=input.k, filter=input.constraints
            )
        elif input.search_type == "similarity_distance_threshold":
            if input.distance_threshold is None:
                raise ValueError("distance_threshold must be provided for " + "similarity_distance_threshold retriever")
            search_res = self.vector_db.similarity_search_by_vector(
                embedding=input.embedding,
                k=input.k,
                distance_threshold=input.distance_threshold,
                filter=input.constraints,
            )
        elif input.search_type == "similarity_score_threshold":
            docs_and_similarities = self.vector_db.similarity_search_with_relevance_scores(
                query=input.text, k=input.k, score_threshold=input.score_threshold, filter=input.constraints
            )
            search_res = [doc for doc, _ in docs_and_similarities]
        elif input.search_type == "mmr":
            search_res = self.vector_db.max_marginal_relevance_search(
                query=input.text,
                k=input.k,
                fetch_k=input.fetch_k,
                lambda_mult=input.lambda_mult,
                filter=input.constraints,
            )
        else:
            raise ValueError(f"{input.search_type} not valid")

        if logflag:
            logger.info(f"retrieve result: {search_res}")

        return search_res


class vCLIPEmbeddings(BaseModel, Embeddings):
    """MeanCLIP Embeddings model."""

    model: Any

    def get_embedding_length(self):
        text_features = self.embed_query("sample_text")
        t_len = len(text_features)
        logger.info(f"text_features: {t_len}")
        return t_len

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that open_clip and torch libraries are installed."""
        try:
            # Use the provided model if present
            if "model" not in values:
                raise ValueError("Model must be provided during initialization.")

        except ImportError:
            raise ImportError("Please ensure CLIP model is loaded")
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model_device = next(self.model.clip.parameters()).device
        text_features = self.model.get_text_embeddings(texts)

        return text_features.detach().numpy()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_video(self, paths: List[str], **kwargs: Any) -> List[List[float]]:
        # Open images directly as PIL images

        video_features = []
        for vid_path in sorted(paths):
            # Encode the video to get the embeddings
            model_device = next(self.model.parameters()).device
            # Preprocess the video for the model
            clip_images = self.load_video_for_vclip(
                vid_path,
                num_frm=self.model.num_frm,
                max_img_size=224,
                start_time=kwargs.get("start_time", None),
                clip_duration=kwargs.get("clip_duration", None),
            )
            embeddings_tensor = self.model.get_video_embeddings([clip_images])

            # Convert tensor to list and add to the video_features list
            embeddings_list = embeddings_tensor.tolist()

            video_features.append(embeddings_list)

        return video_features

    def load_video_for_vclip(self, vid_path, num_frm=4, max_img_size=224, **kwargs):
        # Load video with VideoReader
        import decord

        decord.bridge.set_bridge("torch")
        vr = VideoReader(vid_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        num_frames = len(vr)
        start_idx = int(fps * kwargs.get("start_time", [0])[0])
        end_idx = start_idx + int(fps * kwargs.get("clip_duration", [num_frames])[0])

        frame_idx = np.linspace(start_idx, end_idx, num=num_frm, endpoint=False, dtype=int)  # Uniform sampling
        clip_images = []

        # read images
        temp_frms = vr.get_batch(frame_idx.astype(int).tolist())
        for idx in range(temp_frms.shape[0]):
            im = temp_frms[idx]  # H W C
            clip_images.append(toPIL(im.permute(2, 0, 1)))

        return clip_images


class vCLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_frm = cfg["num_frm"]
        self.model_name = cfg["model_name"]

        self.clip = CLIPModel.from_pretrained(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def get_text_embeddings(self, texts):
        """Input is list of texts."""
        text_inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
        text_features = self.clip.get_text_features(**text_inputs)
        return text_features

    def get_image_embeddings(self, images):
        """Input is list of images."""
        image_inputs = self.processor(images=images, return_tensors="pt")
        image_features = self.clip.get_image_features(**image_inputs)
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
        return torch_cat(vid_embs, dim=0)
