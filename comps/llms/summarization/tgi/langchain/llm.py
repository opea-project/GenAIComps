# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import aiofiles
import docx2txt
from fastapi.responses import StreamingResponse
from huggingface_hub import AsyncInferenceClient
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

from comps import CustomLogger, GeneratedDoc, LLMParamsDoc, ServiceType, opea_microservices, register_microservice

logger = CustomLogger("llm_docsum")
logflag = os.getenv("LOGFLAG", False)

llm_endpoint = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")
llm = AsyncInferenceClient(
    model=llm_endpoint,
    timeout=600,
)


def read_pdf(file):
    loader = PyPDFLoader(file)
    docs = loader.load_and_split()
    return docs


def read_text_from_file(file, save_file_name):
    # read text file
    if file.headers["content-type"] == "text/plain":
        file.file.seek(0)
        content = file.file.read().decode("utf-8")
        # Split text
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(content)
        # Create multiple documents
        file_content = [Document(page_content=t) for t in texts]
    # read pdf file
    elif file.headers["content-type"] == "application/pdf":
        file_content = read_pdf(save_file_name)
    # read docx file
    elif file.headers["content-type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        file_content = docx2txt.process(file)

    return file_content


templ_en = """Write a concise summary of the following:


"{text}"


CONCISE SUMMARY:"""

templ_zh = """请简要概括以下内容:


"{text}"


概况:"""


@register_microservice(
    name="opea_service@llm_docsum",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/docsum",
    host="0.0.0.0",
    port=9000,
)
async def llm_generate(input: LLMParamsDoc):
    if logflag:
        logger.info(input)

    if input.file:
        file_path = f"/tmp/{input.file.filename}"
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(await input.file.read())
        input.query = read_text_from_file(input.file, file_path)
        os.remove(file_path)

    if input.language in ["en", "auto"]:
        templ = templ_en
    elif input.language in ["zh"]:
        templ = templ_zh
    else:
        raise NotImplementedError('Please specify the input language in "en", "zh", "auto"')

    prompt_template = PromptTemplate.from_template(templ)
    prompt = prompt_template.format(text=input.query)

    if logflag:
        logger.info("After prompting:")
        logger.info(prompt)

    text_generation = await llm.text_generation(
        prompt=prompt,
        stream=input.streaming,
        max_new_tokens=input.max_tokens,
        repetition_penalty=input.repetition_penalty,
        temperature=input.temperature,
        top_k=input.top_k,
        top_p=input.top_p,
        typical_p=input.typical_p,
    )

    if input.streaming:

        async def stream_generator():
            chat_response = ""
            async for text in text_generation:
                chat_response += text
                chunk_repr = repr(text.encode("utf-8"))
                if logflag:
                    logger.info(f"[ docsum - text_summarize ] chunk:{chunk_repr}")
                yield f"data: {chunk_repr}\n\n"
            if logflag:
                logger.info(f"[ docsum - text_summarize ] stream response: {chat_response}")
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        if logflag:
            logger.info(text_generation)
        return GeneratedDoc(text=text_generation, prompt=input.query)


if __name__ == "__main__":
    opea_microservices["opea_service@llm_docsum"].start()
