# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import pytest
from opea_haystack.embedders.tei import OPEATextEmbedder

from . import MockBackend

logger = logging.getLogger(__name__)


class TestOPEATextEmbedder:
    def test_init_default(self):
        embedder = OPEATextEmbedder()

        assert embedder.api_url == "http://localhost:6006"
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_init_with_parameters(self):
        embedder = OPEATextEmbedder(
            api_url="https://my-custom-base-url.com",
            prefix="prefix",
            suffix="suffix",
        )
        assert embedder.api_url == "https://my-custom-base-url.com"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"

    def test_to_dict(self):
        component = OPEATextEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "opea_haystack.embedders.tei.text_embedder.OPEATextEmbedder",
            "init_parameters": {
                "api_url": "http://localhost:6006",
                "prefix": "",
                "suffix": "",
                "truncate": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = OPEATextEmbedder(
            api_url="https://my-custom-base-url.com",
            prefix="prefix",
            suffix="suffix",
        )
        data = component.to_dict()
        assert data == {
            "type": "opea_haystack.embedders.tei.text_embedder.OPEATextEmbedder",
            "init_parameters": {
                "api_url": "https://my-custom-base-url.com",
                "prefix": "prefix",
                "suffix": "suffix",
                "truncate": None,
            },
        }

    def test_run_wrong_input_format(self):
        embedder = OPEATextEmbedder()
        embedder.warm_up()
        embedder.backend = MockBackend()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="OPEATextEmbedder expects a string or list of strings as an input."):
            embedder.run(text=list_integers_input)

    @pytest.mark.skipif(
        not os.environ.get("OPEA_EMBEDDING_ENDPOINT_URL", None),
        reason="Export an env var called OPEA_EMBEDDING_ENDPOINT_URL containing the OPEA embedding endpoint url to run this test.",
    )
    @pytest.mark.integration
    def test_run(self):

        embedder = OPEATextEmbedder(prefix="prefix ", suffix=" suffix")
        embedder.warm_up()

        results = embedder.run(text="The food was delicious")

        assert all(isinstance(x, float) for embedding in results["embedding"] for x in embedding)

        # assert "text" in result["meta"]["model"] and "ada" in result["meta"]["model"], (
        #     "The model name does not contain 'text' and 'ada'"
        # )

        assert results["meta"]["usage"] == {"prompt_tokens": 8, "total_tokens": 8}, "Usage information does not match"
