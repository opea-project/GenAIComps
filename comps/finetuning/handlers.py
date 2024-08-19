# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import random
import time
import uuid
from typing import Any, Dict, List, Set

from envs import CHECK_JOB_STATUS_INTERVAL, DATASET_BASE_PATH, MODEL_CONFIG_FILE_MAP, ray_client
from finetune_config import FinetuneConfig
from pydantic_yaml import parse_yaml_raw_as, to_yaml_file
from ray.job_submission import JobSubmissionClient

from comps.cores.proto.api_protocol import FineTuningJob, FineTuningJobIDRequest, FineTuningJobList, FineTuningJobsRequest

FineTuningJobID = str
running_finetuning_jobs: Dict[FineTuningJobID, FineTuningJob] = {}
finetuning_job_to_ray_job: Dict[FineTuningJobID, str] = {}


def handle_create_finetuning_jobs(request: FineTuningJobsRequest):
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

    if request.hyperparameters is not None:
        if request.hyperparameters.epochs != "auto":
            finetune_config.Training.epochs = request.hyperparameters.epochs

        if request.hyperparameters.batch_size != "auto":
            finetune_config.Training.batch_size = request.hyperparameters.batch_size

        if request.hyperparameters.learning_rate_multiplier != "auto":
            finetune_config.Training.learning_rate = request.hyperparameters.learning_rate_multiplier

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

    return job


def handle_list_finetuning_jobs():
    finetuning_jobs_list = FineTuningJobList(data=list(running_finetuning_jobs.values()), has_more=False)

    return finetuning_jobs_list


def handle_retrieve_finetuning_job(request: FineTuningJobIDRequest):
    fine_tuning_job_id = request.fine_tuning_job_id

    job = running_finetuning_jobs.get(fine_tuning_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job '{fine_tuning_job_id}' not found!")
    return job


def handle_cancel_finetuning_job(request: FineTuningJobIDRequest):
    fine_tuning_job_id = request.fine_tuning_job_id

    ray_job_id = finetuning_job_to_ray_job.get(fine_tuning_job_id)
    if ray_job_id is None:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job '{fine_tuning_job_id}' not found!")

    global ray_client
    ray_client = JobSubmissionClient() if ray_client is None else ray_client
    ray_client.stop_job(ray_job_id)

    job = running_finetuning_jobs.get(fine_tuning_job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job with ID '{fine_tuning_job_id}' not found in running jobs!")

    # Check the job status before attempting to cancel
    if job.status == "running":
        # Stop the Ray job
        ray_client.stop_job(ray_job_id)
        # Update job status to cancelled
        job.status = "cancelled"
    else:
        # If the job is not running, return a message indicating it cannot be cancelled
        raise HTTPException(status_code=400, detail=f"Job with ID '{fine_tuning_job_id}' is not running and cannot be cancelled.")
    
    return job