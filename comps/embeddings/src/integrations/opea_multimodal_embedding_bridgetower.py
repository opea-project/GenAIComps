# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import requests
from fastapi.responses import JSONResponse

from comps import OpeaComponent, CustomLogger, ServiceType

from comps import (
    CustomLogger,
    EmbedDoc,
    EmbedMultimodalDoc,
    MultimodalDoc,
    ServiceType,
    TextDoc,
    TextImageDoc,
)

logger = CustomLogger("opea_multimodal_embedding_bridgetower")
logflag = os.getenv("LOGFLAG", False)


class OpeaMultimodalEmbeddingBrigeTower(OpeaComponent):
    """
    A specialized embedding component derived from OpeaComponent for local deployed BrigeTower multimodal embedding services.

    Attributes:
        model_name (str): The name of the embedding model used.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.EMBEDDING.name.lower(), description, config)
        self.url = os.getenv("MMEI_EMBEDDING_HOST_ENDPOINT", "http://0.0.0.0")
        self.port = os.getenv("MMEI_EMBEDDING_PORT_ENDPOINT", "8080")
        self.endpoint = os.getenv("MMEI_EMBEDDING_PATH_ENDPOINT", "/v1/encode")

        self.mmei_embedding_endpoint = os.getenv("MMEI_EMBEDDING_ENDPOINT", f"{self.url}:{self.port}{self.endpoint}")

    async def invoke(self, input: MultimodalDoc) -> EmbedDoc:
        """
        Invokes the embedding service to generate embeddings for the provided input.

        Args:
            input (Union[str, List[str]]): The input text(s) for which embeddings are to be generated.

        Returns:
            List[List[float]]: A list of embedding vectors for the input text(s).
        """
        json = {}
        if isinstance(input, TextDoc):
            json["text"] = input.text
        elif isinstance(input, TextImageDoc):
            json["text"] = input.text.text
            img_bytes = input.image.url.load_bytes()
            base64_img = base64.b64encode(img_bytes).decode("utf-8")
            json["img_b64_str"] = base64_img
        # call multimodal embedding endpoint
        try:
            response = requests.post(self.mmei_embedding_endpoint, headers={"Content-Type": "application/json"}, json=json)
            if response.status_code != 200:
                return JSONResponse(status_code=503, content={"message": "Multimodal embedding endpoint failed!"})

            response_json = response.json()
            embed_vector = response_json["embedding"]
            if isinstance(input, TextDoc):
                res = EmbedDoc(text=input.text, embedding=embed_vector)
            elif isinstance(input, TextImageDoc):
                res = EmbedMultimodalDoc(text=input.text.text, url=input.image.url, embedding=embed_vector)
        except requests.exceptions.ConnectionError:
            res = JSONResponse(status_code=503, content={"message": "Multimodal embedding endpoint not started!"})
        return res

    async def check_health(self) -> bool:
        """Check the health of the microservice by making a GET request to /v1/health_check."""
        try:
            response = requests.get(f"http://{self.url}:{self.port}/v1/health_check")
            if response.status_code == 200:
                return True
            self.logger.error("Health check failed with status code: {response.status_code}")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Health check exception: {e}")
            return False
