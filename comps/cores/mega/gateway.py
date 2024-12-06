# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import os
from io import BytesIO

import requests
from fastapi import Request
from PIL import Image

from ..proto.api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from .constants import MegaServiceEndpoint, ServiceRoleType, ServiceType
from .micro_service import MicroService


def read_pdf(file):
    from langchain.document_loaders import PyPDFLoader

    loader = PyPDFLoader(file)
    docs = loader.load_and_split()
    return docs


def read_text_from_file(file, save_file_name):
    import docx2txt
    from langchain.text_splitter import CharacterTextSplitter

    # read text file
    if file.headers["content-type"] == "text/plain":
        file.file.seek(0)
        content = file.file.read().decode("utf-8")
        # Split text
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(content)
        # Create multiple documents
        file_content = texts
    # read pdf file
    elif file.headers["content-type"] == "application/pdf":
        documents = read_pdf(save_file_name)
        file_content = [doc.page_content for doc in documents]
    # read docx file
    elif (
        file.headers["content-type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or file.headers["content-type"] == "application/octet-stream"
    ):
        file_content = docx2txt.process(save_file_name)

    return file_content


class Gateway:
    def __init__(
        self,
        megaservice,
        host="0.0.0.0",
        port=8888,
        endpoint=str(MegaServiceEndpoint.CHAT_QNA),
        input_datatype=ChatCompletionRequest,
        output_datatype=ChatCompletionResponse,
    ):
        self.megaservice = megaservice
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.input_datatype = input_datatype
        self.output_datatype = output_datatype
        self.service = MicroService(
            self.__class__.__name__,
            service_role=ServiceRoleType.MEGASERVICE,
            service_type=ServiceType.GATEWAY,
            host=self.host,
            port=self.port,
            endpoint=self.endpoint,
            input_datatype=self.input_datatype,
            output_datatype=self.output_datatype,
        )
        self.define_routes()
        self.service.start()

    def define_routes(self):
        self.service.app.router.add_api_route(self.endpoint, self.handle_request, methods=["POST"])
        self.service.app.router.add_api_route(str(MegaServiceEndpoint.LIST_SERVICE), self.list_service, methods=["GET"])
        self.service.app.router.add_api_route(
            str(MegaServiceEndpoint.LIST_PARAMETERS), self.list_parameter, methods=["GET"]
        )

    def add_route(self, endpoint, handler, methods=["POST"]):
        self.service.app.router.add_api_route(endpoint, handler, methods=methods)

    def stop(self):
        self.service.stop()

    async def handle_request(self, request: Request):
        raise NotImplementedError("Subclasses must implement this method")

    def list_service(self):
        response = {}
        for node, service in self.megaservice.services.items():
            # Check if the service has a 'description' attribute and it is not None
            if hasattr(service, "description") and service.description:
                response[node] = {"description": service.description}
            # Check if the service has an 'endpoint' attribute and it is not None
            if hasattr(service, "endpoint") and service.endpoint:
                if node in response:
                    response[node]["endpoint"] = service.endpoint
                else:
                    response[node] = {"endpoint": service.endpoint}
            # If neither 'description' nor 'endpoint' is available, add an error message for the node
            if node not in response:
                response[node] = {"error": f"Service node {node} does not have 'description' or 'endpoint' attribute."}
        return response

    def list_parameter(self):
        pass

    def _handle_message(self, messages):
        images = []
        if isinstance(messages, str):
            prompt = messages
        else:
            messages_dict = {}
            system_prompt = ""
            prompt = ""
            for message in messages:
                msg_role = message["role"]
                if msg_role == "system":
                    system_prompt = message["content"]
                elif msg_role == "user":
                    if type(message["content"]) == list:
                        text = ""
                        text_list = [item["text"] for item in message["content"] if item["type"] == "text"]
                        text += "\n".join(text_list)
                        image_list = [
                            item["image_url"]["url"] for item in message["content"] if item["type"] == "image_url"
                        ]
                        if image_list:
                            messages_dict[msg_role] = (text, image_list)
                        else:
                            messages_dict[msg_role] = text
                    else:
                        messages_dict[msg_role] = message["content"]
                elif msg_role == "assistant":
                    messages_dict[msg_role] = message["content"]
                else:
                    raise ValueError(f"Unknown role: {msg_role}")

            if system_prompt:
                prompt = system_prompt + "\n"
            for role, message in messages_dict.items():
                if isinstance(message, tuple):
                    text, image_list = message
                    if text:
                        prompt += role + ": " + text + "\n"
                    else:
                        prompt += role + ":"
                    for img in image_list:
                        # URL
                        if img.startswith("http://") or img.startswith("https://"):
                            response = requests.get(img)
                            image = Image.open(BytesIO(response.content)).convert("RGBA")
                            image_bytes = BytesIO()
                            image.save(image_bytes, format="PNG")
                            img_b64_str = base64.b64encode(image_bytes.getvalue()).decode()
                        # Local Path
                        elif os.path.exists(img):
                            image = Image.open(img).convert("RGBA")
                            image_bytes = BytesIO()
                            image.save(image_bytes, format="PNG")
                            img_b64_str = base64.b64encode(image_bytes.getvalue()).decode()
                        # Bytes
                        else:
                            img_b64_str = img

                        images.append(img_b64_str)
                else:
                    if message:
                        prompt += role + ": " + message + "\n"
                    else:
                        prompt += role + ":"
        if images:
            return prompt, images
        else:
            return prompt

