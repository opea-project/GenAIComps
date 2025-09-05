# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os

from fastapi import HTTPException

from comps import CustomLogger
from comps.cores.mega.constants import MCPFuncType
from comps.cores.mega.micro_service import opea_microservices, register_microservice
from comps.cores.storages.models import ChatFeedback, FeedbackData, FeedbackId
from comps.cores.storages.stores import get_store_name
from comps.feedback_management.src.integrations.data_store import delete, get, save_or_update

logger = CustomLogger(f"feedback_{get_store_name()}")
logflag = os.getenv("LOGFLAG", False)

# Enable MCP support based on environment variable
enable_mcp = os.getenv("ENABLE_MCP", "").strip().lower() in {"true", "1", "yes"}


@register_microservice(
    name=f"opea_service@feedback_{get_store_name()}",
    endpoint="/v1/feedback/create",
    host="0.0.0.0",
    input_datatype=FeedbackData,
    port=6016,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Create or update feedback data for AI-generated responses including ratings and comments",
)
async def create_feedback_data(feedback: ChatFeedback):
    """Creates and stores a feedback data in database.

    Args:
        feedback (ChatFeedback): The ChatFeedback class object containing feedback data to be stored.

    Returns:
        response (str/bool): FeedbackId of the object created in database. True if data update successfully.
    """
    if logflag:
        logger.info(feedback)

    try:
        response = await save_or_update(feedback)

        if logflag:
            logger.info(response)
        return response

    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@register_microservice(
    name=f"opea_service@feedback_{get_store_name()}",
    endpoint="/v1/feedback/get",
    host="0.0.0.0",
    input_datatype=FeedbackId,
    port=6016,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Retrieve feedback data by ID or get all feedback for a specific user",
)
async def get_feedback(feedback: FeedbackId):
    """Retrieves feedback_data from feedback store based on provided FeedbackId or user.

    Args:
        feedback (FeedbackId): The FeedbackId object containing user and feedback_id or chat_id.

    Returns:
        JSON: Retrieved feedback data if successful, error otherwise.
    """
    if logflag:
        logger.info(feedback)

    try:
        response = await get(feedback)

        if logflag:
            logger.info(response)

        return response

    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@register_microservice(
    name=f"opea_service@feedback_{get_store_name()}",
    endpoint="/v1/feedback/delete",
    host="0.0.0.0",
    input_datatype=FeedbackId,
    port=6016,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Delete specific feedback data by user ID and feedback ID",
)
async def delete_feedback(feedback: FeedbackId):
    """Delete a feedback data from feedback store by given feedback Id.

    Args:
        feedback (FeedbackId): The FeedbackId object containing user and feedback_id or chat_id

    Returns:
        Result of deletion if successful, None otherwise.
    """
    if logflag:
        logger.info(feedback)

    try:
        response = await delete(feedback)

        if logflag:
            logger.info(response)

        return response

    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    opea_microservices[f"opea_service@feedback_{get_store_name()}"].start()
