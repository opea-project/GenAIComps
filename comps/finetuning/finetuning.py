# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from comps import opea_microservices, register_microservice

from comps.cores.proto.api_protocol import FineTuningJobsRequest

from handlers import (
    handle_create_finetuning_jobs,
)

@register_microservice(name="opea_service@finetuning", endpoint="/v1/fine_tuning/jobs", host="0.0.0.0", port=8001)
def create_finetuning_jobs(request: FineTuningJobsRequest):
    return handle_create_finetuning_jobs(request)
