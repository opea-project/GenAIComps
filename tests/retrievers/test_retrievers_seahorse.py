# Copyright (C) 2026 Dnotitia
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class FakeSearchMode:
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class FakeHTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeOpeaComponent:
    def __init__(self, name, type, description, config=None):
        self.name = name
        self.type = type
        self.description = description
        self.config = config or {}


class FakeOpeaComponentRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(component_class):
            cls._registry[name] = component_class
            return component_class

        return decorator


class FakeEmbeddingClient:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class FakeVectorStore:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def health(self):
        return None


class FakeTextSplitter:
    def __init__(self, *args, **kwargs):
        pass

    def split_text(self, content):
        return [content]


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def import_retriever_module(env):
    package_name = "testpkg_retriever"
    module_name = f"{package_name}.seahorse"
    config_name = f"{package_name}.config"

    for name in list(sys.modules):
        if name.startswith(package_name):
            sys.modules.pop(name, None)

    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.HTTPException = FakeHTTPException

    fake_requests = types.ModuleType("requests")
    fake_requests.get = MagicMock()

    fake_langchain_community = _package("langchain_community")
    fake_langchain_community_embeddings = types.ModuleType("langchain_community.embeddings")
    fake_langchain_community_embeddings.HuggingFaceInferenceAPIEmbeddings = FakeEmbeddingClient

    fake_langchain_huggingface = types.ModuleType("langchain_huggingface")
    fake_langchain_huggingface.HuggingFaceEmbeddings = FakeEmbeddingClient

    fake_seahorse_module = types.ModuleType("seahorse_vector_store")
    fake_seahorse_module.SeahorseVectorStore = FakeVectorStore
    fake_seahorse_module.SearchMode = FakeSearchMode

    fake_comps = types.ModuleType("comps")
    fake_comps.CustomLogger = lambda name: FakeLogger()
    fake_comps.EmbedDoc = object
    fake_comps.OpeaComponent = FakeOpeaComponent
    fake_comps.OpeaComponentRegistry = FakeOpeaComponentRegistry
    fake_comps.ServiceType = SimpleNamespace(RETRIEVER=SimpleNamespace(name="RETRIEVER"))

    config_module = types.ModuleType(config_name)
    config_module.EMBED_MODEL = env.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    config_module.HF_TOKEN = env.get("HF_TOKEN", "")
    config_module.SEAHORSE_API_KEY = env.get("SEAHORSE_API_KEY", "")
    config_module.SEAHORSE_BASE_URL = env.get("SEAHORSE_BASE_URL", "")
    config_module.SEAHORSE_EMBEDDING_MODE = env.get("SEAHORSE_EMBEDDING_MODE", "builtin")
    config_module.SEAHORSE_SEARCH_MODE = env.get("SEAHORSE_SEARCH_MODE", "hybrid")
    config_module.TEI_EMBEDDING_ENDPOINT = env.get("TEI_EMBEDDING_ENDPOINT", "")

    retriever_package = _package(package_name)
    retriever_package.config = config_module

    module_path = Path(__file__).resolve().parents[2] / "comps" / "retrievers" / "src" / "integrations" / "seahorse.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name

    with patch.dict(
        sys.modules,
        {
            "fastapi": fake_fastapi,
            "requests": fake_requests,
            "langchain_community": fake_langchain_community,
            "langchain_community.embeddings": fake_langchain_community_embeddings,
            "langchain_huggingface": fake_langchain_huggingface,
            "seahorse_vector_store": fake_seahorse_module,
            "comps": fake_comps,
            package_name: retriever_package,
            config_name: config_module,
            module_name: module,
        },
        clear=False,
    ):
        spec.loader.exec_module(module)
    return module


class TestSeahorseRetriever(unittest.TestCase):
    def test_init_logs_error_when_health_check_fails(self):
        module = import_retriever_module(
            {
                "SEAHORSE_BASE_URL": "https://example.com",
                "SEAHORSE_API_KEY": "secret",
                "SEAHORSE_EMBEDDING_MODE": "builtin",
                "SEAHORSE_SEARCH_MODE": "dense",
            }
        )

        vectorstore = MagicMock()
        vectorstore.health.side_effect = RuntimeError("unreachable")

        with patch.object(module, "SeahorseVectorStore", return_value=vectorstore):
            instance = module.OpeaSeahorseRetriever("OPEA_RETRIEVER_SEAHORSE", "test")
            self.assertIsNotNone(instance)

    def test_init_raises_when_search_mode_unknown(self):
        module = import_retriever_module(
            {
                "SEAHORSE_BASE_URL": "https://example.com",
                "SEAHORSE_API_KEY": "secret",
                "SEAHORSE_EMBEDDING_MODE": "builtin",
                "SEAHORSE_SEARCH_MODE": "not_a_real_mode",
            }
        )

        vectorstore = MagicMock()
        vectorstore.health.return_value = None

        with patch.object(module, "SeahorseVectorStore", return_value=vectorstore):
            with self.assertRaises(RuntimeError) as ctx:
                module.OpeaSeahorseRetriever("OPEA_RETRIEVER_SEAHORSE", "test")
        self.assertIn("Unknown SEAHORSE_SEARCH_MODE", str(ctx.exception))

    def test_invalid_search_type_raises_http_exception(self):
        module = import_retriever_module(
            {
                "SEAHORSE_BASE_URL": "https://example.com",
                "SEAHORSE_API_KEY": "secret",
                "SEAHORSE_EMBEDDING_MODE": "builtin",
                "SEAHORSE_SEARCH_MODE": "dense",
            }
        )

        vectorstore = MagicMock()
        vectorstore.health.return_value = None
        vectorstore.similarity_search.return_value = ["unexpected"]

        with patch.object(module, "SeahorseVectorStore", return_value=vectorstore):
            retriever = module.OpeaSeahorseRetriever("OPEA_RETRIEVER_SEAHORSE", "test")

        with self.assertRaises(module.HTTPException):
            asyncio.run(
                retriever.invoke(
                    SimpleNamespace(text="hello", embedding=[0.1], search_type="unsupported_search", k=2),
                )
            )

    def test_builtin_dense_supports_distance_threshold(self):
        module = import_retriever_module(
            {
                "SEAHORSE_BASE_URL": "https://example.com",
                "SEAHORSE_API_KEY": "secret",
                "SEAHORSE_EMBEDDING_MODE": "builtin",
                "SEAHORSE_SEARCH_MODE": "dense",
            }
        )

        doc_a = object()
        doc_b = object()
        vectorstore = MagicMock()
        vectorstore.health.return_value = None
        vectorstore.similarity_search_with_score.return_value = [(doc_a, 0.15), (doc_b, 0.35)]
        vectorstore.similarity_search.return_value = [doc_b]

        with patch.object(module, "SeahorseVectorStore", return_value=vectorstore):
            retriever = module.OpeaSeahorseRetriever("OPEA_RETRIEVER_SEAHORSE", "test")

        result = asyncio.run(
            retriever.invoke(
                SimpleNamespace(
                    text="hello",
                    embedding=[0.1],
                    search_type="similarity_distance_threshold",
                    distance_threshold=0.2,
                    k=2,
                )
            )
        )

        self.assertEqual(result, [doc_a])

    def test_external_mode_requires_non_empty_embedding(self):
        module = import_retriever_module(
            {
                "SEAHORSE_BASE_URL": "https://example.com",
                "SEAHORSE_API_KEY": "secret",
                "SEAHORSE_EMBEDDING_MODE": "external",
                "SEAHORSE_SEARCH_MODE": "dense",
                "TEI_EMBEDDING_ENDPOINT": "https://tei.example.com",
                "HF_TOKEN": "hf-token",
            }
        )

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"model_id": "bge-small"}

        vectorstore = MagicMock()
        vectorstore.health.return_value = None

        with (
            patch.object(module, "requests") as mock_requests,
            patch.object(module, "HuggingFaceInferenceAPIEmbeddings", return_value=object()),
            patch.object(module, "SeahorseVectorStore", return_value=vectorstore),
        ):
            mock_requests.get.return_value = response
            retriever = module.OpeaSeahorseRetriever("OPEA_RETRIEVER_SEAHORSE", "test")

        with self.assertRaises(module.HTTPException):
            asyncio.run(
                retriever.invoke(
                    SimpleNamespace(text="hello", embedding=[], search_type="similarity", k=2),
                )
            )


if __name__ == "__main__":
    unittest.main()
