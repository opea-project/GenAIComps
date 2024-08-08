# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uvicorn
from fastapi import BackgroundTasks, Cookie, FastAPI, Form, Header, Response
from handlers import (
    handle_cancel_finetuning_job,
    handle_create_finetuning_jobs,
    handle_list_finetuning_jobs,
    handle_retrieve_finetuning_job,
)
from models import FineTuningJob, FineTuningJobList, FineTuningJobsRequest
from pydantic import BaseModel

app = FastAPI()


@app.post("/v1/fine_tuning/jobs", response_model=FineTuningJob)
def create_finetuning_jobs(request: FineTuningJobsRequest, background_tasks: BackgroundTasks):
    return handle_create_finetuning_jobs(request, background_tasks)
    # return {
    #     "object": "fine_tuning.job",
    #     "id": "ftjob-abc123",
    #     "model": "davinci-002",
    #     "created_at": 1692661014,
    #     "finished_at": 1692661190,
    #     "fine_tuned_model": "ft:davinci-002:my-org:custom_suffix:7q8mpxmy",
    #     "organization_id": "org-123",
    #     "result_files": ["file-abc123"],
    #     "status": "succeeded",
    #     "validation_file": None,
    #     "training_file": "file-abc123",
    #     "hyperparameters": {
    #         "n_epochs": 4,
    #         "batch_size": 1,
    #         "learning_rate_multiplier": 1.0,
    #     },
    #     "trained_tokens": 5768,
    #     "integrations": [],
    #     "seed": 0,
    #     "estimated_finish": 0,
    # }


@app.get("/v1/fine_tuning/jobs", response_model=FineTuningJobList)
def list_finetuning_jobs():
    return handle_list_finetuning_jobs()
    # return {
    #     "object": "list",
    #     "data": [
    #         {
    #     "object": "fine_tuning.job",
    #     "id": "ftjob-abc123",
    #     "model": "davinci-002",
    #     "created_at": 1692661014,
    #     "finished_at": 1692661190,
    #     "fine_tuned_model": "ft:davinci-002:my-org:custom_suffix:7q8mpxmy",
    #     "organization_id": "org-123",
    #     "result_files": ["file-abc123"],
    #     "status": "succeeded",
    #     "training_file": "file-abc123",
    #     "hyperparameters": {
    #         "n_epochs": 4,
    #         "batch_size": 1,
    #         "learning_rate_multiplier": 1.0,
    #     },
    #     "trained_tokens": 5768,
    #     "integrations": [],
    #     "seed": 0,
    #     "estimated_finish": 0,
    # },
    #     ],
    #     "has_more": True,
    # }


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
