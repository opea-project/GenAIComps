# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple

import requests

REQUEST_TIMEOUT = 60


class OPEABackend:
    def __init__(
        self,
        api_url: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        self.session = requests.Session()
        self.session.headers.update(headers)

        self.api_url = api_url
        self.model_kwargs = model_kwargs or {}

    def embed(self, inputs: List[str]) -> Tuple[List[List[float]], Dict[str, Any]]:
        url = f"{self.api_url}/embeddings"
        try:
            res = self.session.post(
                url,
                json={
                    "input": inputs,
                    **self.model_kwargs,
                },
                timeout=REQUEST_TIMEOUT,
            )
            res.raise_for_status()
        except requests.HTTPError as e:
            msg = f"Failed to query embedding endpoint: Error - {e.response.text}"
            raise ValueError(msg) from e

        result = res.json()

        embedding = [item["embedding"] for item in result["data"]]

        del result["data"]
        del result["object"]
        meta = result
        return embedding, meta

    def generate(self, prompt: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        url = f"{self.api_url}/v1/chat/completions"

        try:
            res = self.session.post(
                url,
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    **self.model_kwargs,
                },
                timeout=REQUEST_TIMEOUT,
            )
            res.raise_for_status()
        except requests.HTTPError as e:
            msg = f"Failed to query chat completion endpoint: Error - {e.response.text}"
            raise ValueError(msg) from e

        completions = res.json()
        choices = completions["choices"]
        # Sort the choices by index, we don't know whether they're out of order or not
        choices.sort(key=lambda c: c["index"])
        replies = []
        meta = []
        for choice in choices:
            message = choice["message"]
            replies.append(message["content"])
            choice_meta = {
                "role": message["role"],
                "usage": {
                    "prompt_tokens": completions["usage"]["prompt_tokens"],
                    "total_tokens": completions["usage"]["total_tokens"],
                },
            }
            # These fields could be null, the others will always be present
            if "finish_reason" in choice:
                choice_meta["finish_reason"] = choice["finish_reason"]
            if "completion_tokens" in completions["usage"]:
                choice_meta["usage"]["completion_tokens"] = completions["usage"]["completion_tokens"]

            meta.append(choice_meta)

        return replies, meta
