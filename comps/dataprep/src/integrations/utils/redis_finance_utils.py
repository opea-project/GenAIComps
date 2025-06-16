# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import uuid

from docling.document_converter import DocumentConverter
from fastapi import HTTPException
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from openai import OpenAI
from tqdm import tqdm

from comps import CustomLogger
from comps.dataprep.src.integrations.utils.redis_kv import RedisKVStore

logger = CustomLogger("redis_dataprep_finance_data")

# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
TEI_EMBEDDING_ENDPOINT = os.getenv("TEI_EMBEDDING_ENDPOINT", "")

# Redis URL
REDIS_URL_VECTOR = os.getenv("REDIS_URL_VECTOR", "redis://localhost:6379/")
REDIS_URL_KV = os.getenv("REDIS_URL_KV", "redis://localhost:6380/")

# LLM config
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8086")
MAX_TOKENS = os.getenv("MAX_TOKENS", 1024)
TEMPERATURE = os.getenv("TEMPERATURE", 0.2)


def get_embedder():
    if TEI_EMBEDDING_ENDPOINT:
        # create embeddings using TEI endpoint service
        # Huggingface API token for TEI embedding endpoint
        HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        if not HF_TOKEN:
            raise HTTPException(
                status_code=400,
                detail="You MUST provide the `HF_TOKEN` when using `TEI_EMBEDDING_ENDPOINT`.",
            )
        embedder = HuggingFaceEndpointEmbeddings(model=TEI_EMBEDDING_ENDPOINT)
    else:
        # create embeddings using local embedding model
        embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return embedder


IS_RELEVANT_PROMPT = """Should this chunk be part of the document?
Chunk: {chunk}
Document Title: {doc_title}
Give your answer in Yes or No. No other words.
"""

METADATA_PROMPT = """\
Read the following document and extract the following metadata:
- Company name: only the name of the company, do not include Company, Inc., Corp., etc.
- Year
- Quarter: can be empty
- Document type

Output in json format. Examples:
```json
{{"company": "Apple", "year": "2020", "quarter": "", "doc_type": "10-K"}}
```
```json
{{"company": "Apple", "year": "2022", "quarter": "Q1", "doc_type": "10-Q"}}
```
```json
{{"company": "Apple", "year": "2024", "quarter": "Q4", "doc_type": "earnings call transcript"}}
```

Here is the document:
{document}

Only output the metadata in json format. Now start!
"""

CHUNK_SUMMARY_PROMPT = """\
You are a financial analyst. You are given a document extracted from a SEC filing. Read the document and summarize it in a few sentences.
Document:
{doc}
Only output your summary.
"""

TABLE_SUMMARY_PROMPT = """\
You are a financial analyst. You are given a table extracted from a SEC filing. Read the table and give it a descriptive title.
If the table is a financial statement, for example, balance sheet, income statement, cash flow statement, statement of operations, or statement of shareholders' equity, you should specify the type of financial statement in the title.

Table:

{table_md}

Only output the table title.
"""

COMPANY_NAME_PROMPT = """\
Here is the list of company names in the knowledge base:
{company_list}

This is the company of interest: {company}

Determine if the company of interest is the same as any of the companies in the knowledge base.
If yes, map the company of interest to the company name in the knowledge base. Output the company name in  {{}}. Example: {{3M}}.
If none of the companies in the knowledge base match the company of interest, output "NONE".
"""


def parse_metadata_json(metadata):
    """
    metadata: str
    """
    if "```json" in metadata:
        metadata = metadata.split("```json")[1]
        metadata = metadata.split("```")[0]
    elif "```" in metadata:
        metadata = metadata.split("```")[1]
        metadata = metadata.split("```")[0]
    metadata = metadata.strip()
    try:
        metadata = json.loads(metadata)
        return metadata
    except:
        logger.info("Error in parsing metadata.")
        return {}


def post_process_text(text: str) -> str:
    text = text.replace("## Table of Contents", "")
    text = text.replace("Table of Contents", "")
    return text


def split_text(text: str) -> list:
    chunks = text.split("##")
    chunks = [chunk for chunk in chunks if chunk]
    return chunks


def generate_answer(prompt):
    """Use vllm endpoint to generate the answer."""
    # send request to vllm endpoint
    client = OpenAI(
        base_url=f"{LLM_ENDPOINT}/v1",
        api_key="token-abc123",
    )

    params = {
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    completion = client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], **params
    )

    # get response
    response = completion.choices[0].message.content
    logger.info(f"LLM Response: {response}")
    return response


def generate_metadata(full_doc):
    # split the full doc into chunks
    chunks = split_text(full_doc)
    metadata_candidates = []
    for chunk in chunks[:10]:
        prompt = METADATA_PROMPT.format(document=chunk)
        metadata = generate_answer(prompt)
        metadata = parse_metadata_json(metadata)
        if metadata:
            if metadata["company"] and metadata["year"]:
                metadata_candidates.append(metadata)
    if metadata_candidates:
        # majority vote
        final_metadata = {}
        for k in metadata_candidates[0].keys():
            values = [metadata[k] for metadata in metadata_candidates]
            final_metadata[k] = max(set(values), key=values.count)
        for k, v in final_metadata.items():
            if isinstance(v, str):
                final_metadata[k] = v.upper()

        title = f"{final_metadata['company']} {final_metadata['year']} {final_metadata['quarter']} {final_metadata['doc_type']}"
        company_year_quarter = f"{final_metadata['company']}_{final_metadata['year']}_{final_metadata['quarter']}"
        final_metadata["doc_title"] = title
        final_metadata["company_year"] = f"{final_metadata['company']}_{final_metadata['year']}"
        final_metadata["company_year_quarter"] = company_year_quarter

        for k, v in final_metadata.items():
            logger.info(f"{k}: {v}")

    else:
        final_metadata = {}
    return final_metadata


def get_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    return tokenizer


def save_full_doc(full_doc: str, metadata: dict):
    logger.info("Saving full doc....")
    kvstore = RedisKVStore(redis_uri=REDIS_URL_KV)
    index_name = get_index_name("full_doc", metadata)
    # get # of tokens for full doc
    tokenizer = get_tokenizer()
    doc_length = len(tokenizer.encode(full_doc))
    kvstore.put(metadata["doc_title"], {"full_doc": full_doc, "doc_length": doc_length, **metadata}, index_name)
    return None


def save_file_source(filename: str, metadata: dict):
    # this should be done after saving company name
    # return True if file source already in file source list
    # return False if file source did not existed and added to file source list
    logger.info("Saving file source....")
    kvstore = RedisKVStore(redis_uri=REDIS_URL_KV)
    index_name = "file_source"
    # get existing file source list from KV store for this company
    file_source_dict = kvstore.get(metadata["company"], index_name)
    if file_source_dict:
        file_source_list = file_source_dict["source"]
        logger.info(f"Found existing file source list: {file_source_list}")
        # check if filename already in file source list
        if filename in file_source_list:
            logger.info("File source already in file source list.")
            return True
        else:
            file_source_list.append(filename)
            kvstore.put(metadata["company"], {"source": file_source_list}, index_name)
            logger.info(f"Added {filename} to file source list.")
    else:
        logger.info("No existing file source list found. Creating new list.")
        kvstore.put(metadata["company"], {"source": [filename]}, index_name)
    return False


def save_company_name(metadata: dict):
    # get existing companies from KV store
    # collection: company_list
    # key: company
    logger.info("[save_company_name] Saving company name....")
    kvstore = RedisKVStore(redis_uri=REDIS_URL_KV)
    company_list_dict = kvstore.get("company", "company_list")
    if company_list_dict:
        company_list = company_list_dict["company"]
        logger.info(f"[save_company_name] Found existing company list: {company_list}")
    else:
        logger.info("[save_company_name] No existing company list found. Creating new list.")
        company_list = []
        company_list.append(metadata["company"])
        kvstore.put("company", {"company": company_list}, "company_list")
        return metadata

    new_company = metadata["company"]
    # decide if new_company already in company_list
    if new_company in company_list:
        logger.info("[save_company_name] Company already in company list.")
        # no need to change metadata["company"]
        pass
    else:
        # use LLM to decide if new_company is alias of existing company
        logger.info("[save_company_name] Use LLM to decide if company is alias of existing company.")
        prompt = COMPANY_NAME_PROMPT.format(company_list=company_list, company=new_company)
        response = generate_answer(prompt)
        if "NONE" in response.upper():
            logger.info(
                f"[save_company_name] Company {new_company} is not in company list. Add {new_company} to company list."
            )
            # add new_company to company_list
            company_list.append(new_company)
            kvstore.put("company", {"company": company_list}, "company_list")
        else:
            existing_company = response.strip("{}").upper()
            logger.info(
                f"[save_company_name] Company is alias of existing company. Map {new_company} to {existing_company}."
            )
            metadata["company"] = existing_company

    return metadata


def get_index_name(doc_type: str, metadata: dict):
    company = metadata["company"]
    if doc_type == "chunks":
        index_name = f"chunks_{company}"
    elif doc_type == "tables":
        index_name = f"tables_{company}"
    elif doc_type == "titles":
        index_name = f"titles_{company}"
    elif doc_type == "full_doc":
        index_name = f"full_doc_{company}"
    else:
        raise ValueError("doc_type should be either chunks, tables, titles, or full_doc.")
    return index_name


def save_doc_title(doc_title: str, metadata: dict):
    logger.info("Saving doc title....")
    index_name = get_index_name("titles", metadata)
    vector_store = get_vectorstore(index_name)
    keys = vector_store.add_texts([doc_title], [metadata])
    return keys


def parse_doc_and_extract_metadata(filename: str):
    # extract pdf or url with docling and convert into markdown full content and tables
    logger.info("[parse_doc_and_extract_metadata] Parsing document, it may take a few minutes....")
    converter = DocumentConverter()
    conv_res = converter.convert(filename)

    # convert full doc to markdown
    logger.info("[parse_doc_and_extract_metadata] Converting to markdown....")
    full_doc = conv_res.document.export_to_markdown()

    # get doc title and metadata with llm
    logger.info("[parse_doc_and_extract_metadata] Extracting metadata....")
    metadata = generate_metadata(full_doc)

    metadata["source"] = filename
    return conv_res, full_doc, metadata


def split_markdown_and_summarize_save(text, metadata):
    logger.info("Splitting markdown and processing chunks....")
    text = post_process_text(text)
    chunks = split_text(text)
    logger.info(f"Number of chunks: {len(chunks)}")

    keys = []
    for chunk in tqdm(chunks):
        if len(chunk) < 100:
            key = save_chunk((chunk, chunk), metadata)
            keys.extend(key)
            continue
        prompt = CHUNK_SUMMARY_PROMPT.format(doc=chunk)
        summary = generate_answer(prompt)
        key = save_chunk((chunk, summary), metadata)
        keys.extend(key)
    return keys


def save_chunk(chunk: tuple, metadata: dict, doc_type="chunks"):
    # chunk: tuple of (content, summary)
    logger.info(f"Saving {doc_type}....")
    assert doc_type in ["chunks", "tables"], "doc_type should be either chunks or tables"

    chunk_content = chunk[0]
    chunk_summary = chunk[1]

    doc_id = str(uuid.uuid4())
    metadata["doc_id"] = doc_id
    metadata["doc_type"] = doc_type
    logger.info(f"Chunk metadata: {metadata}")
    index_name = get_index_name(doc_type, metadata)
    logger.info(f"Index name: {index_name}")
    logger.info("Embedding summary and saving to vector db....")
    # embed summary and save to vector db
    vector_store = get_vectorstore(index_name)
    key = vector_store.add_texts([chunk_summary], [metadata])

    # save chunk_content to kvstore
    logger.info("Saving to kvstore....")
    kvstore = RedisKVStore(redis_uri=REDIS_URL_KV)
    kvstore.put(doc_id, {"content": chunk_content, "summary": chunk_summary, "metadata": metadata}, index_name)
    return key


def get_table_summary_with_llm(table_md):
    """
    table_md: str
    args: including llm_endpoint_url and model
    """
    prompt = TABLE_SUMMARY_PROMPT.format(table_md=table_md)
    table_summary = generate_answer(prompt)
    logger.info(f"Table summary:\n{table_summary}")
    return table_summary


def process_tables(conv_res, metadata):
    logger.info("Processing tables....")
    keys = []
    for table in tqdm(conv_res.document.tables):
        table_md = table.export_to_markdown()
        context = get_table_summary_with_llm(table_md)
        key = save_chunk((table_md, context), metadata, doc_type="tables")
        keys.extend(key)
    return keys


def post_process_html(full_doc, doc_title):
    logger.info("Post processing extracted webpage....")
    final_doc = ""
    chunks = split_text(full_doc)
    for chunk in tqdm(chunks):
        prompt = IS_RELEVANT_PROMPT.format(chunk=chunk, doc_title=doc_title)
        llm_res = generate_answer(prompt)
        if "YES" in llm_res.upper():
            final_doc += f"##{chunk}"
    return final_doc


def get_company_list():
    kvstore = RedisKVStore(redis_uri=REDIS_URL_KV)
    company_list_dict = kvstore.get("company", "company_list")
    if company_list_dict:
        company_list = company_list_dict["company"]
        return company_list
    else:
        return []


def get_vectorstore(index_name):
    config = RedisConfig(
        index_name=index_name,
        redis_url=REDIS_URL_VECTOR,
        metadata_schema=[
            {"name": "company", "type": "text"},
            {"name": "year", "type": "text"},
            {"name": "quarter", "type": "text"},
            {"name": "doc_type", "type": "text"},
            {"name": "doc_title", "type": "text"},
            {"name": "doc_id", "type": "text"},
            {"name": "company_year", "type": "text"},
            {"name": "company_year_quarter", "type": "text"},
        ],
    )
    embedder = get_embedder()
    vector_store = RedisVectorStore(embedder, config=config)
    return vector_store
