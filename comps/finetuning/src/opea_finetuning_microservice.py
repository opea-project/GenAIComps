# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import base64
import io
import os

from fastapi import Depends, UploadFile
from pydantic import BaseModel

from comps import CustomLogger, opea_microservices, register_microservice
from comps.cores.mega.constants import MCPFuncType
from comps.cores.proto.api_protocol import FineTuningJobIDRequest, UploadFileRequest
from comps.finetuning.src.integrations.finetune_config import FineTuningParams
from comps.finetuning.src.integrations.native import OpeaFinetuning, upload_file
from comps.finetuning.src.integrations.xtune import XtuneFinetuning
from comps.finetuning.src.opea_finetuning_loader import OpeaFinetuningLoader

logger = CustomLogger("opea_finetuning_microservice")
enable_mcp = os.getenv("ENABLE_MCP", "").strip().lower() in {"true", "1", "yes"}

finetuning_component_name = os.getenv("FINETUNING_COMPONENT_NAME", "OPEA_FINETUNING")
# Initialize OpeaComponentLoader
loader = OpeaFinetuningLoader(
    finetuning_component_name,
    description=f"OPEA FINETUNING Component: {finetuning_component_name}",
)


class UploadTrainingFileMCPRequest(BaseModel):
    filename: str
    content_base64: str
    purpose: str = "fine-tune"


@register_microservice(
    name="opea_service@finetuning",
    endpoint="/v1/fine_tuning/jobs",
    host="0.0.0.0",
    port=8015,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Create a fine-tuning job.",
)
def create_finetuning_jobs(request: FineTuningParams):
    return loader.create_finetuning_jobs(request, None)


@register_microservice(
    name="opea_service@finetuning",
    endpoint="/v1/fine_tuning/jobs",
    host="0.0.0.0",
    port=8015,
    methods=["GET"],
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="List fine-tuning jobs.",
)
def list_finetuning_jobs():
    return loader.list_finetuning_jobs()


@register_microservice(
    name="opea_service@finetuning",
    endpoint="/v1/fine_tuning/jobs/retrieve",
    host="0.0.0.0",
    port=8015,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Retrieve a fine-tuning job by ID.",
)
def retrieve_finetuning_job(request: FineTuningJobIDRequest):
    job = loader.retrieve_finetuning_job(request)
    return job


@register_microservice(
    name="opea_service@finetuning",
    endpoint="/v1/fine_tuning/jobs/cancel",
    host="0.0.0.0",
    port=8015,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Cancel a fine-tuning job by ID.",
)
def cancel_finetuning_job(request: FineTuningJobIDRequest):
    job = loader.cancel_finetuning_job(request)
    return job


@register_microservice(
    name="opea_service@finetuning",
    endpoint="/v1/files",
    host="0.0.0.0",
    port=8015,
    enable_mcp=False,
)
async def upload_training_files(request: UploadFileRequest = Depends(upload_file)):
    uploadFileInfo = await loader.upload_training_files(request)
    return uploadFileInfo


@register_microservice(
    name="opea_service@finetuning",
    endpoint="/v1/files/base64",
    host="0.0.0.0",
    port=8015,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Upload a training file using base64-encoded content.",
)
async def upload_training_files_mcp(request: UploadTrainingFileMCPRequest):
    file_bytes = base64.b64decode(request.content_base64)
    upload_file_obj = UploadFile(filename=request.filename, file=io.BytesIO(file_bytes))
    upload_request = UploadFileRequest(purpose=request.purpose, file=upload_file_obj)
    uploadFileInfo = await loader.upload_training_files(upload_request)
    return uploadFileInfo


@register_microservice(
    name="opea_service@finetuning",
    endpoint="/v1/finetune/list_checkpoints",
    host="0.0.0.0",
    port=8015,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="List checkpoints for a fine-tuning job.",
)
def list_checkpoints(request: FineTuningJobIDRequest):
    checkpoints = loader.list_finetuning_checkpoints(request)
    return checkpoints


if __name__ == "__main__":
    opea_microservices["opea_service@finetuning"].start()
