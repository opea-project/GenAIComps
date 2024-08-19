# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uvicorn
from fastapi import BackgroundTasks, FastAPI
from comps.finetuning.handlers import (
    handle_cancel_finetuning_job,
    handle_create_finetuning_jobs,
    handle_list_finetuning_jobs,
    handle_retrieve_finetuning_job,
)

from comps.finetuning.models import FineTuningJob, FineTuningJobList, FineTuningJobsRequest

app = FastAPI()


@app.post("/v1/fine_tuning/jobs", response_model=FineTuningJob)
def create_finetuning_jobs(request: FineTuningJobsRequest, background_tasks: BackgroundTasks):
    return handle_create_finetuning_jobs(request, background_tasks)


@app.get("/v1/fine_tuning/jobs", response_model=FineTuningJobList)
def list_finetuning_jobs():
    return handle_list_finetuning_jobs()


@app.get("/v1/fine_tuning/jobs/{fine_tuning_job_id}", response_model=FineTuningJob)
def retrieve_finetuning_job(fine_tuning_job_id):
    job = handle_retrieve_finetuning_job(fine_tuning_job_id)
    return job


@app.post("/v1/fine_tuning/jobs/{fine_tuning_job_id}/cancel", response_model=FineTuningJob)
def cancel_finetuning_job(fine_tuning_job_id):
    job = handle_cancel_finetuning_job(fine_tuning_job_id)
    return job


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
