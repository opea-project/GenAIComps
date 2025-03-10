# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import time
from typing import List, Optional, Union

from fastapi import Body, File, UploadFile
from integrations.milvus_multimodal import OpeaMultimodalMilvusDataprep
from integrations.redis_multimodal import OpeaMultimodalRedisDataprep
from integrations.vdms_multimodal import OpeaMultimodalVdmsDataprep
from opea_dataprep_loader import OpeaDataprepMultiModalLoader

from comps import (
    CustomLogger,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.dataprep.src.utils import create_upload_folder

logger = CustomLogger("opea_dataprep_multimodal_microservice")
logflag = os.getenv("LOGFLAG", False)
upload_folder = "./uploaded_files/"

dataprep_component_name = os.getenv("DATAPREP_COMPONENT_NAME", "OPEA_DATAPREP_MULTIMODALVDMS")
# Initialize OpeaComponentLoader
loader = OpeaDataprepMultiModalLoader(
    dataprep_component_name,
    description=f"OPEA DATAPREP Multimodal Component: {dataprep_component_name}",
)


@register_microservice(
    name="opea_service@dataprep_multimodal",
    service_type=ServiceType.DATAPREP,
    endpoint="/v1/dataprep/ingest",
    host="0.0.0.0",
    port=5000,
)
@register_statistics(names=["opea_service@dataprep_multimodal"])
async def ingest_files(files: Optional[Union[UploadFile, List[UploadFile]]] = File(None)):
    start = time.time()

    if logflag:
        logger.info(f"[ ingest ] files:{files}")

    try:
        # Use the loader to invoke the component
        response = await loader.ingest_files(files)
        # Log the result if logging is enabled
        if logflag:
            logger.info(f"[ ingest ] Output generated: {response}")
        # Record statistics
        statistics_dict["opea_service@dataprep_multimodal"].append_latency(time.time() - start, None)
        return response
    except Exception as e:
        logger.error(f"Error during dataprep ingest files invocation: {e}")
        raise


@register_microservice(
    name="opea_service@dataprep_multimodal",
    service_type=ServiceType.DATAPREP,
    endpoint="/v1/dataprep/ingest_videos",
    host="0.0.0.0",
    port=5000,
)
@register_statistics(names=["opea_service@dataprep_multimodal"])
async def ingest_videos(files: Optional[Union[UploadFile, List[UploadFile]]] = File(None)):
    start = time.time()

    if logflag:
        logger.info(f"[ ingest ] files:{files}")

    try:
        # Use the loader to invoke the component
        response = await loader.ingest_videos(files)
        # Log the result if logging is enabled
        if logflag:
            logger.info(f"[ ingest ] Output generated: {response}")
        # Record statistics
        statistics_dict["opea_service@dataprep_multimodal"].append_latency(time.time() - start, None)
        return response
    except Exception as e:
        logger.error(f"Error during dataprep ingest videos invocation: {e}")
        raise


@register_microservice(
    name="opea_service@dataprep_multimodal",
    service_type=ServiceType.DATAPREP,
    endpoint="/v1/dataprep/generate_transcripts",
    host="0.0.0.0",
    port=5000,
)
@register_statistics(names=["opea_service@dataprep_multimodal"])
async def ingest_generate_transcripts(files: Optional[Union[UploadFile, List[UploadFile]]] = File(None)):
    start = time.time()

    if logflag:
        logger.info(f"[ ingest ] files:{files}")
    try:
        # Use the loader to invoke the component
        response = await loader.ingest_generate_transcripts(files)
        # Log the result if logging is enabled
        if logflag:
            logger.info(f"[ ingest ] Output generated: {response}")
        # Record statistics
        statistics_dict["opea_service@dataprep_multimodal"].append_latency(time.time() - start, None)
        return response
    except Exception as e:
        logger.error(f"Error during dataprep generate_transcripts invocation: {e}")
        raise


@register_microservice(
    name="opea_service@dataprep_multimodal",
    service_type=ServiceType.DATAPREP,
    endpoint="/v1/dataprep/generate_captions",
    host="0.0.0.0",
    port=5000,
)
@register_statistics(names=["opea_service@dataprep_multimodal"])
async def ingest_generate_captions(files: Optional[Union[UploadFile, List[UploadFile]]] = File(None)):
    start = time.time()

    if logflag:
        logger.info(f"[ ingest ] files:{files}")

    try:
        # Use the loader to invoke the component
        response = await loader.ingest_generate_captions(files)
        # Log the result if logging is enabled
        if logflag:
            logger.info(f"[ ingest ] Output generated: {response}")
        # Record statistics
        statistics_dict["opea_service@dataprep_multimodal"].append_latency(time.time() - start, None)
        return response
    except Exception as e:
        logger.error(f"Error during dataprep generate_captions invocation: {e}")
        raise


@register_microservice(
    name="opea_service@dataprep_multimodal",
    service_type=ServiceType.DATAPREP,
    endpoint="/v1/dataprep/get",
    host="0.0.0.0",
    port=5000,
)
@register_statistics(names=["opea_service@dataprep_multimodal"])
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
        statistics_dict["opea_service@dataprep_multimodal"].append_latency(time.time() - start, None)
        return response
    except Exception as e:
        logger.error(f"Error during dataprep get files invocation: {e}")
        raise


@register_microservice(
    name="opea_service@dataprep_multimodal",
    service_type=ServiceType.DATAPREP,
    endpoint="/v1/dataprep/get/{filename}",
    host="0.0.0.0",
    port=5000,
    methods=["GET"],
)
@register_statistics(names=["opea_service@dataprep_multimodal"])
async def get_one_file(filename: str):
    start = time.time()

    if logflag:
        logger.info("[ get ] start to get ingested files")

    try:
        # Use the loader to invoke the component
        response = await loader.get_one_file(filename)
        # Log the result if logging is enabled
        if logflag:
            logger.info(f"[ get ] ingested files: {response}")
        # Record statistics
        statistics_dict["opea_service@dataprep_multimodal"].append_latency(time.time() - start, None)
        return response
    except Exception as e:
        logger.error(f"Error during dataprep get one file invocation: {e}")
        raise


@register_microservice(
    name="opea_service@dataprep_multimodal",
    service_type=ServiceType.DATAPREP,
    endpoint="/v1/dataprep/get_videos",
    host="0.0.0.0",
    port=5000,
    methods=["GET"],
)
@register_statistics(names=["opea_service@dataprep_multimodal"])
async def get_videos():
    start = time.time()

    if logflag:
        logger.info("[ get ] start to get ingested files")

    try:
        # Use the loader to invoke the component
        response = await loader.get_videos()
        # Log the result if logging is enabled
        if logflag:
            logger.info(f"[ get ] ingested files: {response}")
        # Record statistics
        statistics_dict["opea_service@dataprep_multimodal"].append_latency(time.time() - start, None)
        return response
    except Exception as e:
        logger.error(f"Error during dataprep get videos invocation: {e}")
        raise


@register_microservice(
    name="opea_service@dataprep_multimodal",
    service_type=ServiceType.DATAPREP,
    endpoint="/v1/dataprep/delete",
    host="0.0.0.0",
    port=5000,
)
@register_statistics(names=["opea_service@dataprep_multimodal"])
async def delete_files(file_path: str = Body(..., embed=True)):
    start = time.time()

    if logflag:
        logger.info("[ delete ] start to delete ingested files")

    try:
        # Use the loader to invoke the component
        response = await loader.delete_files(file_path)
        # Log the result if logging is enabled
        if logflag:
            logger.info(f"[ delete ] deleted result: {response}")
        # Record statistics
        statistics_dict["opea_service@dataprep_multimodal"].append_latency(time.time() - start, None)
        return response
    except Exception as e:
        logger.error(f"Error during dataprep delete invocation: {e}")
        raise


if __name__ == "__main__":
    logger.info("OPEA Dataprep Multimodal Microservice is starting...")
    create_upload_folder(upload_folder)
    opea_microservices["opea_service@dataprep_multimodal"].start()
