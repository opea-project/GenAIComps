# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import urllib.parse

from fastapi import BackgroundTasks, File, UploadFile
from typing import List, Optional, Union
from comps import opea_microservices, register_microservice
from comps.cores.proto.api_protocol import FineTuningJobIDRequest, FineTuningJobsRequest
from comps.finetuning.handlers import (
    handle_cancel_finetuning_job,
    handle_create_finetuning_jobs,
    handle_list_finetuning_jobs,
    handle_retrieve_finetuning_job,
    save_content_to_local_disk,
    handle_list_finetuning_checkpoints,
    DATASET_BASE_PATH,
)


@register_microservice(name="opea_service@finetuning", endpoint="/v1/fine_tuning/jobs", host="0.0.0.0", port=8001)
def create_finetuning_jobs(request: FineTuningJobsRequest, background_tasks: BackgroundTasks):
    return handle_create_finetuning_jobs(request, background_tasks)


@register_microservice(
    name="opea_service@finetuning", endpoint="/v1/fine_tuning/jobs", host="0.0.0.0", port=8001, methods=["GET"]
)
def list_finetuning_jobs():
    return handle_list_finetuning_jobs()


@register_microservice(
    name="opea_service@finetuning", endpoint="/v1/fine_tuning/jobs/retrieve", host="0.0.0.0", port=8001
)
def retrieve_finetuning_job(request: FineTuningJobIDRequest):
    job = handle_retrieve_finetuning_job(request)
    return job


@register_microservice(
    name="opea_service@finetuning", endpoint="/v1/fine_tuning/jobs/cancel", host="0.0.0.0", port=8001
)
def cancel_finetuning_job(request: FineTuningJobIDRequest):
    job = handle_cancel_finetuning_job(request)
    return job


@register_microservice(
    name="opea_service@finetuning", endpoint="/v1/finetune/upload_training_files", host="0.0.0.0", port=8001,
)
async def upload_training_files(files: Optional[Union[UploadFile, List[UploadFile]]] = File(None),):
    if files:
        if not isinstance(files, list):
            files = [files]
        for file in files:
            filename = urllib.parse.quote(file.filename, safe="")
            save_path = os.path.join(DATASET_BASE_PATH, filename)
            await save_content_to_local_disk(save_path, file)

    return {"status": 200, "message": "Training files uploaded."}


@register_microservice(
    name="opea_service@finetuning", endpoint="/v1/finetune/list_checkpoints", host="0.0.0.0", port=8001
)
def list_checkpoints(request: FineTuningJobIDRequest):
    checkpoints = handle_list_finetuning_checkpoints(request)
    return {"status": 200, "checkpoints":str(checkpoints)}


if __name__ == "__main__":
    opea_microservices["opea_service@finetuning"].start()
