# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os

from comps import CustomLogger, ServiceType
from comps.cores.mega.constants import MCPFuncType
from comps.cores.mega.micro_service import opea_microservices, register_microservice
from comps.cores.storages.models import PromptCreate, PromptId
from comps.cores.storages.stores import get_store_name
from comps.prompt_registry.src.integrations.data_store import delete, get, save

logger = CustomLogger(f"prompt_registry_{get_store_name()}")
logflag = os.getenv("LOGFLAG", False)
enable_mcp = os.getenv("ENABLE_MCP", "").strip().lower() in {"true", "1", "yes"}


@register_microservice(
    name=f"opea_service@prompt_registry_{get_store_name()}",
    service_type=ServiceType.PROMPT_REGISTRY,
    endpoint="/v1/prompt/create",
    host="0.0.0.0",
    input_datatype=PromptCreate,
    port=6018,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Store a user's preferred prompt in the database with metadata",
)
async def create_prompt(prompt: PromptCreate):
    """Creates and stores a prompt in prompt store.

    Args:
        prompt (PromptCreate): The PromptCreate class object containing the data to be stored.

    Returns:
        JSON (PromptResponse): PromptResponse class object, None otherwise.
    """
    if logflag:
        logger.info(prompt)
    try:
        response = await save(prompt)
        if logflag:
            logger.info(response)
        return response

    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")
        return None


@register_microservice(
    name=f"opea_service@prompt_registry_{get_store_name()}",
    service_type=ServiceType.PROMPT_REGISTRY,
    endpoint="/v1/prompt/get",
    host="0.0.0.0",
    input_datatype=PromptId,
    port=6018,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Retrieve prompts by user, ID, or keyword search from the database",
)
async def get_prompt(prompt: PromptId):
    """Retrieves prompt from prompt store based on provided PromptId or user.

    Args:
        prompt (PromptId): The PromptId object containing user and prompt_id.

    Returns:
        JSON: Retrieved prompt data if successful, None otherwise.
    """
    if logflag:
        logger.info(prompt)
    try:
        response = await get(prompt)
        if logflag:
            logger.info(response)
        return response

    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")
        return None


@register_microservice(
    name=f"opea_service@prompt_registry_{get_store_name()}",
    service_type=ServiceType.PROMPT_REGISTRY,
    endpoint="/v1/prompt/delete",
    host="0.0.0.0",
    input_datatype=PromptId,
    port=6018,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Delete a prompt by ID from the database",
)
async def delete_prompt(prompt: PromptId):
    """Delete a prompt from prompt store by given PromptId.

    Args:
        prompt (PromptId): The PromptId object containing user and prompt_id.

    Returns:
        Result of deletion if successful, None otherwise.
    """
    if logflag:
        logger.info(prompt)
    try:
        response = await delete(prompt)
        if logflag:
            logger.info(response)
        return response

    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")
        return None


if __name__ == "__main__":
    # Start the unified service with all endpoints
    opea_microservices[f"opea_service@prompt_registry_{get_store_name()}"].start()
