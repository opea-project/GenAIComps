# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from handlers import (
    handle_cancel_finetuning_job,
    handle_create_finetuning_jobs,
    handle_list_finetuning_jobs,
    handle_retrieve_finetuning_job,
)

from comps import opea_microservices, register_microservice
from comps.cores.proto.api_protocol import FineTuningJobIDRequest, FineTuningJobListRequest, FineTuningJobsRequest


@register_microservice(name="opea_service@finetuning", endpoint="/v1/fine_tuning/jobs", host="0.0.0.0", port=8001)
def create_finetuning_jobs(request: FineTuningJobsRequest):
    return handle_create_finetuning_jobs(request)


@register_microservice(
    name="opea_service@finetuning", endpoint="/v1/fine_tuning/jobs", host="0.0.0.0", port=8001, methods=["GET"]
)
def list_finetuning_jobs(request: FineTuningJobListRequest):
    return handle_list_finetuning_jobs(request)


@register_microservice(
    name="opea_service@finetuning",
    endpoint="/v1/fine_tuning/jobs/{fine_tuning_job_id}",
    host="0.0.0.0",
    port=8001,
    methods=["GET"],
)
def retrieve_finetuning_job(request: FineTuningJobIDRequest):
    job = handle_retrieve_finetuning_job(request)
    return job


@register_microservice(
    name="opea_service@finetuning",
    endpoint="/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel",
    host="0.0.0.0",
    port=8001,
)
def cancel_finetuning_job(request: FineTuningJobIDRequest):
    job = handle_cancel_finetuning_job(request)
    return job


if __name__ == "__main__":
    opea_microservices["opea_service@finetuning"].start()
