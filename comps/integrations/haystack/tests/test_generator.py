# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from haystack import default_from_dict, default_to_dict
from haystack.utils import Secret
from opea_haystack.generators import OPEAGenerator
from requests_mock import Mocker


@pytest.fixture
def mock_local_chat_completion(requests_mock: Mocker) -> None:
    requests_mock.post(
        "http://localhost:9009/v1/chat/completions",
        json={
            "choices": [
                {
                    "message": {"content": "Hello!", "role": "system"},
                    "usage": {"prompt_tokens": 3, "total_tokens": 5, "completion_tokens": 9},
                    "finish_reason": "stop",
                    "index": 0,
                },
                {
                    "message": {"content": "How are you?", "role": "system"},
                    "usage": {"prompt_tokens": 3, "total_tokens": 5, "completion_tokens": 9},
                    "finish_reason": "stop",
                    "index": 1,
                },
            ],
            "usage": {
                "prompt_tokens": 3,
                "total_tokens": 5,
                "completion_tokens": 9,
            },
        },
    )


class TestOPEAGenerator:
    def test_init_default(self):
        generator = OPEAGenerator()

        assert generator._api_url == "http://localhost:9009"
        assert generator._model_arguments == {}

    def test_init_with_parameters(self):
        generator = OPEAGenerator(
            api_url="http://myurl:9009/v1",
            model_arguments={
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
                "seed": None,
                "bad": None,
                "stop": None,
            },
        )
        assert generator._api_url == "http://myurl:9009/v1"
        assert generator._model_arguments == {
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
            "seed": None,
            "bad": None,
            "stop": None,
        }

    def test_to_dict(self):
        init_parameters = {
            "api_url": "http://localhost:9009/v1",
            "model_arguments": {},
        }
        generator = OPEAGenerator()
        data = default_to_dict(generator, **init_parameters)
        assert data == {
            "type": "opea_haystack.generators.generator.OPEAGenerator",
            "init_parameters": init_parameters,
        }

    @pytest.mark.skipif(
        not os.environ.get("OPEA_GENERATOR_ENDPOINT_URL", None),
        reason="Export an env var called OPEA_GENERATOR_ENDPOINT_URL containing the URL to call.",
    )
    @pytest.mark.integration
    def test_run_integration_with_opea_backend(self):
        url = os.environ["OPEA_GENERATOR_ENDPOINT_URL"]
        generator = OPEAGenerator(
            api_url=url,
            model_arguments={
                "temperature": 0.2,
            },
        )
        generator.warm_up()
        result = generator.run(prompt="What is the answer?")

        assert result["replies"]
        assert result["meta"]

    @pytest.mark.integration
    @pytest.mark.usefixtures("mock_local_chat_completion")
    def test_run_integration_with_default_model_opea_backend(self):
        url = "http://localhost:9009"
        generator = OPEAGenerator(
            api_url=url,
            model_arguments={
                "temperature": 0.2,
            },
        )
        generator.warm_up()
        result = generator.run(prompt="What is the answer?")

        assert result["replies"]
        assert result["meta"]
