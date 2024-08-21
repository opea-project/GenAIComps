# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import BackgroundTasks

from comps import opea_microservices, register_microservice
from comps.cores.proto.api_protocol import FineTuningJobIDRequest, FineTuningJobsRequest
from comps.finetuning.handlers import (
    handle_cancel_finetuning_job,
    handle_create_finetuning_jobs,
    handle_list_finetuning_jobs,
    handle_retrieve_finetuning_job,
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


if __name__ == "__main__":
    opea_microservices["opea_service@finetuning"].start()
