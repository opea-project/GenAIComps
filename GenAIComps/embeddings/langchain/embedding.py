import asyncio
from http_service import HTTPService
from _docarray import TextDoc, EmbedDoc1024
from docarray.base_doc import DocArrayResponse
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


async def setup():
    runtime_args = {
        'title': 'test_service',
        'description': "this is a test.",
        'protocol': 'http',
        'port': 8099,
        'host': 'localhost',
    }
    # breakpoint()
    service = HTTPService(runtime_args=runtime_args, cors=False)
    app = service.app

    @app.post(
        path="/v1/embed",
        response_model=EmbedDoc1024,
        response_class=DocArrayResponse,
        summary="Get the embeded vector of the input text",
        tags=["Debug"],
    )
    def embedding(input: TextDoc) -> EmbedDoc1024:
        embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        embed_vector = embeddings.embed_query(input.text)
        res = EmbedDoc1024(text=input.text, embedding=embed_vector)
        return res

    await service.initialize_server()
    await service.execute_server()


asyncio.run(setup())