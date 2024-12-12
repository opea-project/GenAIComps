# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import time
from typing import List, Optional, Union
from fastapi import File, Form, UploadFile
from integrations.redis_dataprep import OpeaRedisDataprep
from opea_dataprep_controller import OpeaDataprepController
from comps import (
    CustomLogger,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)


logger = CustomLogger("prepare_doc_redis")
logflag = os.getenv("LOGFLAG", False)
# Initialize Controller
controller = OpeaDataprepController()


# Register components
try:
    # Instantiate OpeaEmbedding and PredictionguardEmbedding components
    opea_embedding = OpeaRedisDataprep(
        name="OpeaRedisDataprep",
        description="OPEA Redis Dataprep Service",
    )
    # Register components with the controller
    controller.register(opea_embedding)
    # Discover and activate a healthy component
    controller.discover_and_activate()
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")


@register_microservice(
    name="opea_service@dataprep",
    service_type=ServiceType.DATAPREP,
    endpoint="/v1/dataprep/ingest",
    host="0.0.0.0",
    port=5000,
)
@register_statistics(names=["opea_service@dataprep"])
async def ingest_files(
    files: Optional[Union[UploadFile, List[UploadFile]]] = File(None),
    link_list: Optional[str] = Form(None),
    chunk_size: int = Form(1500),
    chunk_overlap: int = Form(100),
    process_table: bool = Form(False),
    table_strategy: str = Form("fast"),
):
    start = time.time()

    if logflag:
        logger.info(f"[ ingest ] files:{files}")
        logger.info(f"[ ingest ] link_list:{link_list}")

    try:
        # Use the controller to invoke the active component
        response = controller.ingest_files(
            files, link_list, chunk_size, chunk_overlap, process_table, table_strategy
        )
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
        logger.info(f"[ get ] start to get ingested files")

    try:
        # Use the controller to invoke the active component
        response = controller.get_files()
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
async def get_files():
    start = time.time()

    if logflag:
        logger.info(f"[ delete ] start to delete ingested files")

    try:
        # Use the controller to invoke the active component
        response = controller.delete_files()
        # Log the result if logging is enabled
        if logflag:
            logger.info(f"[ delete ] deleted result: {response}")
        # Record statistics
        statistics_dict["opea_service@dataprep"].append_latency(time.time() - start, None)
        return response
    except Exception as e:
        logger.error(f"Error during dataprep delete invocation: {e}")
        raise


if __name__ == "__main__":
    logger.info("OPEA Dataprep Microservice is starting...")
    opea_microservices["opea_service@dataprep"].start()
