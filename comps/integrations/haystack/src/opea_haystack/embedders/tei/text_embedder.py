# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

from haystack import component, default_to_dict
from opea_haystack.utils import OPEABackend

from .truncate import EmbeddingTruncateMode

_DEFAULT_API_URL = "http://localhost:6006"


@component
class OPEATextEmbedder:
    """
    A component for embedding strings using embedding models provided by
    [OPEA](https://opea.dev).

    For models that differentiate between query and document inputs,
    this component embeds the input string as a query.

    Usage example:
    ```python
    from opea_haystack.embedders.tei import OPEATextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = OPEATextEmbedder(api_url="http://localhost:6006")
    text_embedder.warm_up()

    print(text_embedder.run(text_to_embed)
    ```
    """

    def __init__(
        self,
        api_url: str = _DEFAULT_API_URL,
        prefix: str = "",
        suffix: str = "",
        truncate: Optional[Union[EmbeddingTruncateMode, str]] = None,
    ):
        """Create an OPEATextEmbedder component.

        :param api_url:
            Custom API URL for the OPEA.
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        :param truncate:
            Specifies how inputs longer that the maximum token length should be truncated.
            If None the behavior is model-dependent, see the official documentation for more information.
        """

        self.api_url = api_url
        self.prefix = prefix
        self.suffix = suffix

        if isinstance(truncate, str):
            truncate = EmbeddingTruncateMode.from_str(truncate)
        self.truncate = truncate

        self.backend: Optional[Any] = None
        self._initialized = False

    def warm_up(self):
        """Initializes the component."""
        if self._initialized:
            return

        model_kwargs = {"input_type": "query"}
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
            truncate=str(self.truncate) if self.truncate is not None else None,
        )

    @component.output_types(embedding=List[float])
    def run(self, text: Union[str, List[str]]):
        """Embed a string or list of strings.

        :param text:
            The text(s) to embed.
        :returns:
            A dictionary with the following keys and values:
            - `embedding` - Embeddngs of the text(s).
        :raises RuntimeError:
            If the component was not initialized.
        :raises TypeError:
            If the input is not a string or list of strings.
        """
        if not self._initialized:
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)
        elif not isinstance(text, (str, list)) or (isinstance(text, list) and not isinstance(text[0], str)):
            msg = (
                "OPEATextEmbedder expects a string or list of strings as an input."
                "In case you want to embed a list of Documents, please use the OPEADocumentEmbedder."
            )
            raise TypeError(msg)
        elif not text:
            msg = "Cannot embed an empty string."
            raise ValueError(msg)

        assert self.backend is not None
        if isinstance(text, str):
            text = [text]

        text_to_embed = [self.prefix + t + self.suffix for t in text]
        embedding, metadata = self.backend.embed(text_to_embed)
        return {"embedding": embedding, "meta": metadata}
