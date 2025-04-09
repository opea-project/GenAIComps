# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import time
from typing import Annotated, List, Optional, Union

from fastapi import Body, Depends, File, Form, HTTPException, UploadFile
from integrations.elasticsearch import OpeaElasticSearchDataprep
from integrations.milvus import OpeaMilvusDataprep
from integrations.neo4j_llamaindex import OpeaNeo4jLlamaIndexDataprep
from integrations.opensearch import OpeaOpenSearchDataprep
from integrations.pgvect import OpeaPgvectorDataprep
from integrations.pipecone import OpeaPineConeDataprep
from integrations.qdrant import OpeaQdrantDataprep
from integrations.redis import OpeaRedisDataprep
from integrations.redis_finance import OpeaRedisDataprepFinance
from integrations.vdms import OpeaVdmsDataprep
from opea_dataprep_loader import OpeaDataprepLoader

from comps import (
    CustomLogger,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.dataprep.src.utils import create_upload_folder
from comps.cores.proto.api_protocol import DataprepRequest, Neo4jDataprepRequest, RedisDataprepRequest

logger = CustomLogger("opea_dataprep_microservice")
logflag = os.getenv("LOGFLAG", False)
upload_folder = "./uploaded_files/"

dataprep_component_name = os.getenv("DATAPREP_COMPONENT_NAME", "OPEA_DATAPREP_REDIS")
# Initialize OpeaComponentLoader
loader = OpeaDataprepLoader(
    dataprep_component_name,
    description=f"OPEA DATAPREP Component: {dataprep_component_name}",
)

from fastapi import Request
async def resolve_dataprep_request(request: Request):
    form = await request.form()
    
    # 检查是否有 Redis 特定参数
    if "index_name" in form:
        logger.info(f"chunk_size: {form.get("chunk_size")}")
        return RedisDataprepRequest(
            files=form.get("files"),
            link_list=form.get("link_list", ),
            chunk_size=form.get("chunk_size"),
            chunk_overlap=form.get("chunk_overlap"),
            process_table=form.get("process_table"),
            table_strategy=form.get("table_strategy"),
            index_name=form.get("index_name"),
        )
    
    # 检查是否有 Neo4j 特定参数
    if "ingest_from_graphDB" in form:
        return Neo4jDataprepRequest(
            files=form.get("files"),
            link_list=form.get("link_list"),
            chunk_size=form.get("chunk_size"),
            chunk_overlap=form.get("chunk_overlap"),
            process_table=form.get("process_table"),
            table_strategy=form.get("table_strategy"),
            ingest_from_graphDB=form.get("ingest_from_graphDB"),
        )
    
    # 默认使用 Base 类
    return DataprepRequest(
        files=form.get("files"),
        link_list=form.get("link_list"),
        chunk_size=form.get("chunk_size"),
        chunk_overlap=form.get("chunk_overlap"),
        process_table=form.get("process_table"),
        table_strategy=form.get("table_strategy"),
    )

@register_microservice(
    name="opea_service@dataprep",
    service_type=ServiceType.DATAPREP,
    endpoint="/v1/dataprep/ingest",
    host="0.0.0.0",
    port=5000,
)
@register_statistics(names=["opea_service@dataprep"])
async def ingest_files(
    input: Union[DataprepRequest, RedisDataprepRequest, Neo4jDataprepRequest] = Depends(resolve_dataprep_request),
    # redis: RedisDataprepRequest = Depends(),
    # neo4j: Neo4jDataprepRequest = Depends(),
    # base: DataprepRequest = Depends(),
    # base: Annotated[Optional[DataprepRequest], Depends()] = None,
    # redis: Annotated[Optional[RedisDataprepRequest], Depends()] = None,
    # neo4j: Annotated[Optional[Neo4jDataprepRequest], Depends()] = None,
):
    
    start = time.time()

    files = input.files
    if isinstance(input, RedisDataprepRequest):
        index_name = input.index_name
        logger.info(f"[ ingest ] Redis mode: index_name={index_name}")
    elif isinstance(input, Neo4jDataprepRequest):
        ingest_from_graphDB = input.ingest_from_graphDB
        logger.info(f"[ ingest ] Neo4j mode: ingest_from_graphDB={ingest_from_graphDB}")
    else:
        link_list = input.link_list
        logger.info(f"[ ingest ] Base mode: link_list={link_list}")

    if logflag:
        logger.info(f"[ ingest ] files: {files}")

    try:
        response = await loader.ingest_files(input=input)

        # Log the result if logging is enabled
        if logflag:
            logger.info(f"[ ingest ] Output generated: {response}")
        # Record statistics
        statistics_dict["opea_service@dataprep"].append_latency(time.time() - start, None)
        return response
    except Exception as e:
        logger.error(f"Error during dataprep ingest invocation: {e}")
        raise


@register_microservice(
    name="opea_service@dataprep",
    service_type=ServiceType.DATAPREP,
    endpoint="/v1/dataprep/get",
    host="0.0.0.0",
    port=5000,
)
@register_statistics(names=["opea_service@dataprep"])
async def get_files():
    start = time.time()

    if logflag:
        logger.info("[ get ] start to get ingested files")

    try:
        # Use the loader to invoke the component
        response = await loader.get_files()
        # Log the result if logging is enabled
        if logflag:
            logger.info(f"[ get ] ingested files: {response}")
        # Record statistics
        statistics_dict["opea_service@dataprep"].append_latency(time.time() - start, None)
        return response
    except Exception as e:
        logger.error(f"Error during dataprep get invocation: {e}")
        raise


@register_microservice(
    name="opea_service@dataprep",
    service_type=ServiceType.DATAPREP,
    endpoint="/v1/dataprep/delete",
    host="0.0.0.0",
    port=5000,
)
@register_statistics(names=["opea_service@dataprep"])
async def delete_files(file_path: str = Body(..., embed=True), index_name: str = Body(None, embed=True)):
    start = time.time()

    if logflag:
        logger.info("[ delete ] start to delete ingested files")

    try:
        # Use the loader to invoke the component
        if dataprep_component_name == "OPEA_DATAPREP_REDIS":
            response = await loader.delete_files(file_path, index_name)
        else:
            if index_name:
                logger.error(
                    'Error during dataprep delete files: "index_name" option is supported if "DATAPREP_COMPONENT_NAME" environment variable is set to "OPEA_DATAPREP_REDIS". i.e: export DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_REDIS"'
                )
                raise
            # Use the loader to invoke the component
            response = await loader.delete_files(file_path)

        # Log the result if logging is enabled
        if logflag:
            logger.info(f"[ delete ] deleted result: {response}")
        # Record statistics
        statistics_dict["opea_service@dataprep"].append_latency(time.time() - start, None)
        return response
    except Exception as e:
        logger.error(f"Error during dataprep delete invocation: {e}")
        raise


@register_microservice(
    name="opea_service@dataprep",
    service_type=ServiceType.DATAPREP,
    endpoint="/v1/dataprep/indices",
    host="0.0.0.0",
    port=5000,
)
@register_statistics(names=["opea_service@dataprep"])
async def get_list_of_indices():
    start = time.time()
    if logflag:
        logger.info("[ get ] start to get list of indices.")

    if dataprep_component_name != "OPEA_DATAPREP_REDIS":
        logger.error(
            'Error during dataprep - get list of indices: "index_name" option is supported if "DATAPREP_COMPONENT_NAME" environment variable is set to "OPEA_DATAPREP_REDIS". i.e: export DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_REDIS"'
        )
        raise

    try:
        # Use the loader to invoke the component
        response = await loader.get_list_of_indices()

        # Log the result if logging is enabled
        if logflag:
            logger.info(f"[ get ] list of indices: {response}")

        # Record statistics
        statistics_dict["opea_service@dataprep"].append_latency(time.time() - start, None)

        return response
    except Exception as e:
        logger.error(f"Error during dataprep get list of indices: {e}")
        raise


if __name__ == "__main__":
    logger.info("OPEA Dataprep Microservice is starting...")
    create_upload_folder(upload_folder)
    opea_microservices["opea_service@dataprep"].start()
