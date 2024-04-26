from docarray import BaseDoc, DocList
from docarray.typing import NdArray


class TextDoc(BaseDoc):
    text: str


class EmbedDoc768(BaseDoc):
    text: str
    embedding: NdArray[768]


class EmbedDoc1024(BaseDoc):
    text: str
    embedding: NdArray[1024]


class GenerateDoc(BaseDoc):
    text: str
    prompt: str