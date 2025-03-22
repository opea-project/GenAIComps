# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_to_dict, logging
from opea_haystack.utils import OPEABackend
from tqdm import tqdm

from .truncate import EmbeddingTruncateMode

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "http://localhost:6006"


@component
class OPEADocumentEmbedder:
    """
    A component for embedding documents using embedding models provided by
    [OPEA](https://opea.dev).

    Usage example:
    ```python
    from haystack import Document
    from opea_haystack.embedders.tei import OPEADocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = OPEADocumentEmbedder(api_url="http://localhost:6006")
    document_embedder.warm_up()

    result = document_embedder.run([doc])
    print(result["documents"][0].embedding)
    ```
    """

    def __init__(
        self,
        api_url: str = _DEFAULT_API_URL,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        truncate: Optional[Union[EmbeddingTruncateMode, str]] = None,
    ):
        """Create an OPEADocumentEmbedder component.

        :param api_url:
            Custom API URL for OPEA.
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        :param batch_size:
            Number of Documents to encode at once.
            Cannot be greater than 50.
        :param progress_bar:
            Whether to show a progress bar or not.
        :param meta_fields_to_embed:
            List of meta fields that should be embedded along with the Document text.
        :param embedding_separator:
            Separator used to concatenate the meta fields to the Document text.
        :param truncate:
            Specifies how inputs longer that the maximum token length should be truncated.
            If None the behavior is model-dependent, see the official documentation for more information.
        """

        self.api_url = api_url
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

        if isinstance(truncate, str):
            truncate = EmbeddingTruncateMode.from_str(truncate)
        self.truncate = truncate

        self.backend: Optional[Any] = None
        self._initialized = False

    def warm_up(self):
        """Initializes the component."""
        if self._initialized:
            return

        model_kwargs = {"input_type": "passage"}
        if self.truncate is not None:
            model_kwargs["truncate"] = str(self.truncate)
        self.backend = OPEABackend(
            api_url=self.api_url,
            model_kwargs=model_kwargs,
        )

        self._initialized = True

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_url=self.api_url,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            truncate=str(self.truncate) if self.truncate is not None else None,
        )

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]
            text_to_embed = (
                self.prefix + self.embedding_separator.join([*meta_values_to_embed, doc.content or ""]) + self.suffix
            )
            texts_to_embed.append(text_to_embed)

        return texts_to_embed

    def _embed_batch(self, texts_to_embed: List[str], batch_size: int) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        meta: Dict[str, Any] = {}

        assert self.backend is not None

        for i in tqdm(
            range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + batch_size]

            sorted_embeddings, metadata = self.backend.embed(batch)
            all_embeddings.extend(sorted_embeddings)

            if "model" not in meta:
                meta["model"] = metadata["model"]
            if "usage" not in meta:
                meta["usage"] = dict(metadata["usage"])
            else:
                meta["usage"]["prompt_tokens"] += metadata["usage"]["prompt_tokens"]
                meta["usage"]["total_tokens"] += metadata["usage"]["total_tokens"]

        return all_embeddings, meta

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """Embed a list of Documents.

        The embedding of each Document is stored in the `embedding` field of the Document.

        :param documents:
            A list of Documents to embed.
        :returns:
            A dictionary with the following keys and values:
            - `documents` - List of processed Documents with embeddings.
            - `meta` - Metadata on usage statistics, etc.
        :raises RuntimeError:
            If the component was not initialized.
        :raises TypeError:
            If the input is not a string.
        """
        if not self._initialized:
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)
        elif not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            msg = (
                "OPEADocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the OPEATextEmbedder."
            )
            raise TypeError(msg)

        for doc in documents:
            if not doc.content:
                logger.warning(f"Document '{doc.id}' has no content to embed.")

        texts_to_embed = self._prepare_texts_to_embed(documents)
        embeddings, metadata = self._embed_batch(texts_to_embed, self.batch_size)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": metadata}
