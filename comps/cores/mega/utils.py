# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import ipaddress
import json
import multiprocessing
import os
import random
from io import BytesIO
from socket import AF_INET, SOCK_STREAM, socket
from typing import List, Optional, Union

import requests
from PIL import Image

from .logger import CustomLogger


def is_port_free(host: str, port: int) -> bool:
    """Check if a given port on a host is free.

    :param host: The host to check.
    :param port: The port to check.
    :return: True if the port is free, False otherwise.
    """
    with socket(AF_INET, SOCK_STREAM) as session:
        return session.connect_ex((host, port)) != 0


def check_ports_availability(host: Union[str, List[str]], port: Union[int, List[int]]) -> bool:
    """Check if one or more ports on one or more hosts are free.

    :param host: The host(s) to check.
    :param port: The port(s) to check.
    :return: True if all ports on all hosts are free, False otherwise.
    """
    hosts = [host] if isinstance(host, str) else host
    ports = [port] if isinstance(port, int) else port

    return all(is_port_free(h, p) for h in hosts for p in ports)


class ConfigError(Exception):
    """Custom exception for configuration errors."""

    pass


def load_model_configs(model_configs: str) -> dict:
    """Load and validate the model configurations .

    If valid, return the configuration for the specified model name.
    """
    logger = CustomLogger("models_loader")
    try:
        configs = json.loads(model_configs)
        if not isinstance(configs, list) or not configs:
            raise ConfigError("MODEL_CONFIGS must be a non-empty JSON array.")
        required_keys = {"model_name", "displayName", "endpoint", "minToken", "maxToken"}
        configs_map = {}
        for config in configs:
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                raise ConfigError(f"Missing required configuration fields: {missing_keys}")
            empty_keys = [key for key in required_keys if not config.get(key)]
            if empty_keys:
                raise ConfigError(f"Empty values found for configuration fields: {empty_keys}")
            model_name = config["model_name"]
            configs_map[model_name] = config
        if not configs_map:
            raise ConfigError("No valid configurations found.")
        return configs_map
    except json.JSONDecodeError:
        logger.error("Error parsing MODEL_CONFIGS environment variable as JSON.")
        raise ConfigError("MODEL_CONFIGS is not valid JSON.")
    except ConfigError as e:
        logger.error(str(e))
        raise


def get_access_token(token_url: str, client_id: str, client_secret: str) -> str:
    """Get access token using OAuth client credentials flow."""
    logger = CustomLogger("tgi_or_tei_service_auth")
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(token_url, data=data, headers=headers)
    if response.status_code == 200:
        token_info = response.json()
        return token_info.get("access_token", "")
    else:
        logger.error(f"Failed to retrieve access token: {response.status_code}, {response.text}")
        return ""


class SafeContextManager:
    """This context manager ensures that the `__exit__` method of the
    sub context is called, even when there is an Exception in the
    `__init__` method."""

    def __init__(self, context_to_manage):
        self.context_to_manage = context_to_manage

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.context_to_manage.__exit__(exc_type, exc_val, exc_tb)


def handle_message(messages):
    images = []
    if isinstance(messages, str):
        prompt = messages
    else:
        messages_list = []
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
                        messages_list.append((msg_role, (text, image_list)))
                    else:
                        messages_list.append((msg_role, text))
                else:
                    messages_list.append((msg_role, message["content"]))
            elif msg_role == "assistant":
                messages_list.append((msg_role, message["content"]))
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        if system_prompt:
            prompt = system_prompt + "\n"
        for role, message in messages_list:
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


def sanitize_env(value: Optional[str]) -> Optional[str]:
    """Remove quotes from a configuration value if present.

    Args:
        value (str): The configuration value to sanitize.
    Returns:
        str: The sanitized configuration value.
    """
    if value is None:
        return None
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    elif value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return value
