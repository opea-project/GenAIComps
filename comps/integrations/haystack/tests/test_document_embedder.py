# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from haystack import Document, default_from_dict
from opea_haystack.embedders.tei import EmbeddingTruncateMode, OPEADocumentEmbedder

from . import MockBackend


class TestOPEADocumentEmbedder:
    def test_init_default(self):
        embedder = OPEADocumentEmbedder()

        assert embedder.api_url == "http://localhost:6006"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_with_parameters(self):
        embedder = OPEADocumentEmbedder(
            api_url="http://localhost:6006/test",
            prefix="prefix",
            suffix="suffix",
            batch_size=30,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )

        assert embedder.api_url == "http://localhost:6006/test"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 30
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    def test_to_dict(self):
        component = OPEADocumentEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "opea_haystack.embedders.tei.document_embedder.OPEADocumentEmbedder",
            "init_parameters": {
                "api_url": "http://localhost:6006",
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "truncate": None,
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        component = OPEADocumentEmbedder(
            api_url="http://localhost:6006/test",
            prefix="prefix",
            suffix="suffix",
            batch_size=10,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
            truncate=EmbeddingTruncateMode.END,
        )
        data = component.to_dict()
        assert data == {
            "type": "opea_haystack.embedders.tei.document_embedder.OPEADocumentEmbedder",
            "init_parameters": {
                "api_url": "http://localhost:6006/test",
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 10,
                "progress_bar": False,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
                "truncate": "END",
            },
        }

    def test_from_dict(self):
        data = {
            "type": "opea_haystack.embedders.tei.document_embedder.OPEADocumentEmbedder",
            "init_parameters": {
                "api_url": "http://localhost:6006/test",
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 10,
                "progress_bar": False,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
                "truncate": "START",
            },
        }
        component = default_from_dict(OPEADocumentEmbedder, data)
        assert component.api_url == "http://localhost:6006/test"
        assert component.prefix == "prefix"
        assert component.suffix == "suffix"
        assert component.batch_size == 10
        assert component.progress_bar is False
        assert component.meta_fields_to_embed == ["test_field"]
        assert component.embedding_separator == " | "
        assert component.truncate == EmbeddingTruncateMode.START

    def test_from_dict_defaults(self):
        data = {
            "type": "opea_haystack.embedders.tei.document_embedder.OPEADocumentEmbedder",
            "init_parameters": {},
        }
        component = default_from_dict(OPEADocumentEmbedder, data)
        assert component.api_url == "http://localhost:6006"
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.batch_size == 32
        assert component.progress_bar
        assert component.meta_fields_to_embed == []
        assert component.embedding_separator == "\n"
        assert component.truncate is None

    def test_prepare_texts_to_embed_w_metadata(self):
        documents = [
            Document(content=f"document number {i}:\ncontent", meta={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder = OPEADocumentEmbedder(
            meta_fields_to_embed=["meta_field"],
            embedding_separator=" | ",
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        # note that newline is replaced by space
        assert prepared_texts == [
            "meta_value 0 | document number 0:\ncontent",
            "meta_value 1 | document number 1:\ncontent",
            "meta_value 2 | document number 2:\ncontent",
            "meta_value 3 | document number 3:\ncontent",
            "meta_value 4 | document number 4:\ncontent",
        ]

    def test_prepare_texts_to_embed_w_suffix(self):
        documents = [Document(content=f"document number {i}") for i in range(5)]

        embedder = OPEADocumentEmbedder(
            prefix="my_prefix ",
            suffix=" my_suffix",
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "my_prefix document number 0 my_suffix",
            "my_prefix document number 1 my_suffix",
            "my_prefix document number 2 my_suffix",
            "my_prefix document number 3 my_suffix",
            "my_prefix document number 4 my_suffix",
        ]

    def test_embed_batch(self):
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]
        embedder = OPEADocumentEmbedder()

        embedder.warm_up()
        embedder.backend = MockBackend()

        embeddings, metadata = embedder._embed_batch(texts_to_embed=texts, batch_size=2)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 3
            assert all(isinstance(x, float) for x in embedding)

        assert metadata == {"model": "my_model", "usage": {"prompt_tokens": 3 * 4, "total_tokens": 3 * 4}}

    def test_run(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        embedder = OPEADocumentEmbedder(
            prefix="prefix ",
            suffix=" suffix",
            meta_fields_to_embed=["topic"],
            embedding_separator=" | ",
        )

        embedder.warm_up()
        embedder.backend = MockBackend()

        result = embedder.run(documents=docs)

        documents_with_embeddings = result["documents"]
        metadata = result["meta"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 3
            assert all(isinstance(x, float) for x in doc.embedding)
        assert metadata == {"model": "my_model", "usage": {"prompt_tokens": 4, "total_tokens": 4}}

    def test_run_custom_batch_size(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        embedder = OPEADocumentEmbedder(
            prefix="prefix ",
            suffix=" suffix",
            meta_fields_to_embed=["topic"],
            embedding_separator=" | ",
            batch_size=1,
        )

        embedder.warm_up()
        embedder.backend = MockBackend()

        result = embedder.run(documents=docs)

        documents_with_embeddings = result["documents"]
        metadata = result["meta"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 3
            assert all(isinstance(x, float) for x in doc.embedding)

        assert metadata == {"model": "my_model", "usage": {"prompt_tokens": 2 * 4, "total_tokens": 2 * 4}}

    def test_run_wrong_input_format(self):
        embedder = OPEADocumentEmbedder()

        embedder.warm_up()
        embedder.backend = MockBackend()

        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="OPEADocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=string_input)

        with pytest.raises(TypeError, match="OPEADocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=list_integers_input)

    def test_run_empty_document(self, caplog):
        embedder = OPEADocumentEmbedder()

        embedder.warm_up()
        embedder.backend = MockBackend()

        # Write check using caplog that a logger.warning is raised
        with caplog.at_level("WARNING"):
            embedder.run(documents=[Document(content="")])
            assert "has no content to embed." in caplog.text

    def test_run_on_empty_list(self):
        embedder = OPEADocumentEmbedder()

        embedder.warm_up()
        embedder.backend = MockBackend()

        empty_list_input = []
        result = embedder.run(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]

    @pytest.mark.skipif(
        not os.environ.get("OPEA_EMBEDDING_ENDPOINT_URL", None),
        reason="Export an env var called OPEA_EMBEDDING_ENDPOINT_URL containing the OPEA embedding endpoint url to run this test.",
    )
    @pytest.mark.integration
    def test_run_integration_with_opea_backend(self):
        url = os.environ["OPEA_EMBEDDING_ENDPOINT_URL"]
        embedder = OPEADocumentEmbedder(
            api_url=url,
        )
        embedder.warm_up()
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        result = embedder.run(docs)
        docs_with_embeddings = result["documents"]

        assert isinstance(docs_with_embeddings, list)
        assert len(docs_with_embeddings) == len(docs)
        for doc in docs_with_embeddings:
            assert isinstance(doc.embedding, list)
            assert isinstance(doc.embedding[0], float)
