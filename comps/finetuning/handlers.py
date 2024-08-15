# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import random
import time
import uuid
from typing import Any, Dict, List, Set

from envs import CHECK_JOB_STATUS_INTERVAL, DATASET_BASE_PATH, MODEL_CONFIG_FILE_MAP, ray_client
from pydantic_yaml import parse_yaml_raw_as, to_yaml_file

from comps.cores.proto.api_protocol import FineTuningJob, FineTuningJobsRequest

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

    # background_tasks.add_task(update_job_status, job.id)

    return job
