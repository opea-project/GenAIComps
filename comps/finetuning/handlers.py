# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import random
import time
import uuid
from typing import Any, Dict, List, Set

from fastapi import BackgroundTasks, HTTPException
from pydantic_yaml import parse_yaml_raw_as, to_yaml_file
from ray.job_submission import JobSubmissionClient
from ray.train.base_trainer import TrainingFailedError
from ray.tune.logger import LoggerCallback

from .llm_on_ray.finetune.finetune import main
from .llm_on_ray.finetune.finetune_config import FinetuneConfig
from .models import FineTuningJob, FineTuningJobEvent, FineTuningJobList, FineTuningJobsRequest

MODEL_CONFIG_FILE_MAP = {
    "meta-llama/Llama-2-7b-chat-hf": "./models/llama-2-7b-chat-hf.yaml",
    "mistralai/Mistral-7B-v0.1": "./models/mistral-7b-v0.1.yaml",
}

DATASET_BASE_PATH = "datasets"

FineTuningJobID = str
CHECK_JOB_STATUS_INTERVAL = 5  # Check every 5 secs

global ray_client
ray_client: JobSubmissionClient = None

running_finetuning_jobs: Dict[FineTuningJobID, FineTuningJob] = {}
finetuning_job_to_ray_job: Dict[FineTuningJobID, str] = {}


# Add a background task to periodicly update job status
def update_job_status(job_id: FineTuningJobID):
    while True:
        job_status = ray_client.get_job_status(finetuning_job_to_ray_job[job_id])
        status = str(job_status).lower()
        # Ray status "stopped" is OpenAI status "cancelled"
        status = "cancelled" if status == "stopped" else status
        print(f"Status of job {job_id} is '{status}'")
        running_finetuning_jobs[job_id].status = status
        if status == "finished" or status == "cancelled" or status == "failed":
            break
        time.sleep(CHECK_JOB_STATUS_INTERVAL)


def handle_create_finetuning_jobs(request: FineTuningJobsRequest, background_tasks: BackgroundTasks):
    base_model = request.model
    train_file = request.training_file
    train_file_path = os.path.join(DATASET_BASE_PATH, train_file)

    model_config_file = MODEL_CONFIG_FILE_MAP.get(base_model)
    if not model_config_file:
        raise HTTPException(status_code=404, detail=f"Base model '{base_model}' not supported!")

    if not os.path.exists(train_file_path):
        raise HTTPException(status_code=404, detail=f"Training file '{train_file}' not found!")

    with open(model_config_file) as f:
        finetune_config = parse_yaml_raw_as(FinetuneConfig, f)

    finetune_config.Dataset.train_file = train_file_path

    job = FineTuningJob(
        id=f"ft-job-{uuid.uuid4()}",
        model=base_model,
        created_at=int(time.time()),
        training_file=train_file,
        hyperparameters={
            "n_epochs": finetune_config.Training.epochs,
            "batch_size": finetune_config.Training.batch_size,
            "learning_rate_multiplier": finetune_config.Training.learning_rate,
        },
        status="running",
        # TODO: Add seed in finetune config
        seed=random.randint(0, 1000),
    )

    finetune_config_file = f"jobs/{job.id}.yaml"
    to_yaml_file(finetune_config_file, finetune_config)

    global ray_client
    ray_client = JobSubmissionClient() if ray_client is None else ray_client

    ray_job_id = ray_client.submit_job(
        # Entrypoint shell command to execute
        entrypoint=f"python finetune_runner.py --config_file {finetune_config_file}",
        # Path to the local directory that contains the script.py file
        runtime_env={"working_dir": "./"},
    )
    print(f"Submitted Ray job: {ray_job_id} ...")

    running_finetuning_jobs[job.id] = job
    finetuning_job_to_ray_job[job.id] = ray_job_id

    background_tasks.add_task(update_job_status, job.id)

    return job


def handle_list_finetuning_jobs():
    finetuning_jobs_list = FineTuningJobList(data=list(running_finetuning_jobs.values()), has_more=False)

    return finetuning_jobs_list


def handle_retrieve_finetuning_job(fine_tuning_job_id):
    job = running_finetuning_jobs.get(fine_tuning_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job '{fine_tuning_job_id}' not found!")
    return job


def handle_cancel_finetuning_job(fine_tuning_job_id):
    ray_job_id = finetuning_job_to_ray_job.get(fine_tuning_job_id)
    if ray_job_id is None:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job '{fine_tuning_job_id}' not found!")

    global ray_client
    ray_client = JobSubmissionClient() if ray_client is None else ray_client
    ray_client.stop_job(ray_job_id)

    job = running_finetuning_jobs.get(fine_tuning_job_id)
    job.status = "cancelled"
    return job


# def cancel_all_jobs():
#     global ray_client
#     ray_client = JobSubmissionClient() if ray_client is None else ray_client
#     # stop all jobs
#     for job_id in finetuning_job_to_ray_job.values():
#         ray_client.stop_job(job_id)

#     for job_id in running_finetuning_jobs:
#         running_finetuning_jobs[job_id].status = "cancelled"
#     return running_finetuning_jobs
