# langchain-opea

This package contains the LangChain integrations for OPENAI Compatible [OPEA](https://opea.dev/) Microservices.

## Installation

You can install LangChain OPEA package in several ways:

### Install from Source

To install the package from the source, run:

```bash
pip install poetry && poetry install --with test
```

### Install from Wheel Package

To install the package from a pre-built wheel, run:

```bash
pip install dist/langchain_opea-0.1.0-py3-none-any.whl
```

### Install from PyPi

> **Note:** Once the package is available on PyPi, you can install it using:

```bash
pip install -U langchain-opea
```

## Chat Models

`ChatOPEA` class exposes OPENAI Compatible chat models from OPEA.

```python
from langchain_opea import ChatOPEA

llm = ChatOPEA(
    model="Intel/neural-chat-7b-v3-3", opea_api_key="my_secret_value", opea_api_base="http://localhost:9009/v1"
)
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`OPEAEmbeddings` class exposes OPENAI Compatible embeddings from OPEA.

```python
from langchain_opea import OPEAEmbeddings

embeddings = OPEAEmbeddings(
    model="BAAI/bge-large-en-v1.5",
    opea_api_key="my_secret_value",
    opea_api_base="http://localhost:6006/v1",
)
embeddings.embed_query("What is the meaning of life?")
```

## LLMs

`OPEALLM` class exposes OPENAI Compatible LLMs from OPEA.

```python
from langchain_opea import OPEALLM

llm = OPEALLM(
    model="Intel/neural-chat-7b-v3-3", opea_api_key="my_secret_value", opea_api_base="http://localhost:9009/v1"
)
llm.invoke("The meaning of life is")
```

Check out [Samples](./samples/README.md) for more examples using the OPEA Langchain package.
