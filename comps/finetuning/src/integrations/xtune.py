# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import random
import re
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Dict

from fastapi import BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic_yaml import to_yaml_file
from ray.job_submission import JobSubmissionClient

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry
from comps.cores.proto.api_protocol import (
    FileObject,
    FineTuningJob,
    FineTuningJobCheckpoint,
    FineTuningJobIDRequest,
    FineTuningJobList,
    UploadFileRequest,
)
from comps.finetuning.src.integrations.finetune_config import FinetuneConfig, FineTuningParams

logger = CustomLogger("opea")

DATASET_BASE_PATH = "datasets"
JOBS_PATH = "jobs"
OUTPUT_DIR = "output"

if not os.path.exists(DATASET_BASE_PATH):
    os.mkdir(DATASET_BASE_PATH)
if not os.path.exists(JOBS_PATH):
    os.mkdir(JOBS_PATH)
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

FineTuningJobID = str
CheckpointID = str
CheckpointPath = str

CHECK_JOB_STATUS_INTERVAL = 5  # Check every 5 secs

global ray_client
ray_client: JobSubmissionClient = None

running_finetuning_jobs: Dict[FineTuningJobID, FineTuningJob] = {}
finetuning_job_to_ray_job: Dict[FineTuningJobID, str] = {}
checkpoint_id_to_checkpoint_path: Dict[CheckpointID, CheckpointPath] = {}


# Add a background task to periodicly update job status
def update_job_status(job_id: FineTuningJobID):
    while True:
        job_status = ray_client.get_job_status(finetuning_job_to_ray_job[job_id])
        status = str(job_status).lower()
        # Ray status "stopped" is OpenAI status "cancelled"
        status = "cancelled" if status == "stopped" else status
        logger.info(f"Status of job {job_id} is '{status}'")
        running_finetuning_jobs[job_id].status = status
        if status == "succeeded" or status == "cancelled" or status == "failed":
            break
        time.sleep(CHECK_JOB_STATUS_INTERVAL)


async def save_content_to_local_disk(save_path: str, content):
    save_path = Path(save_path)
    try:
        if isinstance(content, str):
            with open(save_path, "w", encoding="utf-8") as file:
                file.write(content)
        else:
            with save_path.open("wb") as fout:
                content = await content.read()
                fout.write(content)
    except Exception as e:
        logger.info(f"Write file failed. Exception: {e}")
        raise Exception(status_code=500, detail=f"Write file {save_path} failed. Exception: {e}")


async def upload_file(purpose: str = Form(...), file: UploadFile = File(...)):
    return UploadFileRequest(purpose=purpose, file=file)


@OpeaComponentRegistry.register("XTUNE_FINETUNING")
class XtuneFinetuning(OpeaComponent):
    """A specialized finetuning component derived from OpeaComponent for finetuning services."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, "finetuning", description, config)

    def create_finetuning_jobs(self, request: FineTuningParams, background_tasks: BackgroundTasks):
        model = request.model
        train_file = request.training_file
        finetune_config = FinetuneConfig(General=request.General)
        if finetune_config.General.xtune_config.device == "XPU":
            flag = 1
        else:
            flag = 0
        if os.getenv("HF_TOKEN", None):
            finetune_config.General.config.token = os.getenv("HF_TOKEN", None)

        job = FineTuningJob(
            id=f"ft-job-{uuid.uuid4()}",
            model=model,
            created_at=int(time.time()),
            training_file=train_file,
            hyperparameters={},
            status="running",
            seed=random.randint(0, 1000) if request.seed is None else request.seed,
        )

        finetune_config_file = f"{JOBS_PATH}/{job.id}.yaml"
        to_yaml_file(finetune_config_file, finetune_config)

        global ray_client
        ray_client = JobSubmissionClient() if ray_client is None else ray_client
        if finetune_config.General.xtune_config.tool == "clip":
            ray_job_id = ray_client.submit_job(
                # Entrypoint shell command to execute
                entrypoint=f"cd integrations/xtune/src/llamafactory/clip_finetune && export DATA={finetune_config.General.xtune_config.dataset_root} && bash scripts/clip_finetune/{finetune_config.General.xtune_config.trainer}.sh {finetune_config.General.xtune_config.dataset} {finetune_config.General.xtune_config.model} 0 {finetune_config.General.xtune_config.device} > /tmp/test.log 2>&1 || true",
            )

        else:
            if flag == 1:
                ray_job_id = ray_client.submit_job(
                    # Entrypoint shell command to execute
                    entrypoint=f"cd integrations/xtune/src/llamafactory/adaclip_finetune && python  train.py --config {finetune_config.General.xtune_config.config_file} --frames_dir {finetune_config.General.xtune_config.dataset_root}{finetune_config.General.xtune_config.dataset}/frames --top_k 16 --freeze_cnn --frame_agg mlp --resume {finetune_config.General.xtune_config.model} --xpu --batch_size 8  > /tmp/test.log 2>&1 || true",
                )
            else:
                ray_job_id = ray_client.submit_job(
                    # Entrypoint shell command to execute
                    entrypoint=f"cd integrations/xtune/src/llamafactory/adaclip_finetune && python  train.py --config {finetune_config.General.config_file} --frames_dir {finetune_config.General.dataset_root}{finetune_config.General.dataset}/frames --top_k 16 --freeze_cnn --frame_agg mlp --resume {finetune_config.General.model}--batch_size 8  > /tmp/test.log 2>&1 || true",
                )

        logger.info(f"Submitted Ray job: {ray_job_id} ...")

        running_finetuning_jobs[job.id] = job
        finetuning_job_to_ray_job[job.id] = ray_job_id

        background_tasks.add_task(update_job_status, job.id)

        return job

    def list_finetuning_jobs(self):
        finetuning_jobs_list = FineTuningJobList(data=list(running_finetuning_jobs.values()), has_more=False)

        return finetuning_jobs_list

    def retrieve_finetuning_job(self, request: FineTuningJobIDRequest):
        fine_tuning_job_id = request.fine_tuning_job_id

        job = running_finetuning_jobs.get(fine_tuning_job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Fine-tuning job '{fine_tuning_job_id}' not found!")
        return job

    def cancel_finetuning_job(self, request: FineTuningJobIDRequest):
        fine_tuning_job_id = request.fine_tuning_job_id

        ray_job_id = finetuning_job_to_ray_job.get(fine_tuning_job_id)
        if ray_job_id is None:
            raise HTTPException(status_code=404, detail=f"Fine-tuning job '{fine_tuning_job_id}' not found!")

        global ray_client
        ray_client = JobSubmissionClient() if ray_client is None else ray_client
        ray_client.stop_job(ray_job_id)

        job = running_finetuning_jobs.get(fine_tuning_job_id)
        job.status = "cancelled"
        return job

    def list_finetuning_checkpoints(self, request: FineTuningJobIDRequest):
        fine_tuning_job_id = request.fine_tuning_job_id

        job = running_finetuning_jobs.get(fine_tuning_job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Fine-tuning job '{fine_tuning_job_id}' not found!")
        output_dir = os.path.join(OUTPUT_DIR, job.id)
        checkpoints = []
        if os.path.exists(output_dir):
            # Iterate over the contents of the directory and add an entry for each
            files = os.listdir(output_dir)
            for file in files:  # Loop over directory contents
                file_path = os.path.join(output_dir, file)
                if os.path.isdir(file_path) and file.startswith("checkpoint"):
                    steps = re.findall("\d+", file)[0]
                    checkpointsResponse = FineTuningJobCheckpoint(
                        id=f"ftckpt-{uuid.uuid4()}",  # Generate a unique ID
                        created_at=int(time.time()),  # Use the current timestamp
                        fine_tuned_model_checkpoint=file_path,  # Directory path itself
                        fine_tuning_job_id=fine_tuning_job_id,
                        object="fine_tuning.job.checkpoint",
                        step_number=steps,
                    )
                    checkpoints.append(checkpointsResponse)
            if job.status == "succeeded":
                checkpointsResponse = FineTuningJobCheckpoint(
                    id=f"ftckpt-{uuid.uuid4()}",  # Generate a unique ID
                    created_at=int(time.time()),  # Use the current timestamp
                    fine_tuned_model_checkpoint=output_dir,  # Directory path itself
                    fine_tuning_job_id=fine_tuning_job_id,
                    object="fine_tuning.job.checkpoint",
                )
                checkpoints.append(checkpointsResponse)

        return checkpoints

    async def upload_training_files(self, request: UploadFileRequest):
        file = request.file
        if file is None:
            raise HTTPException(status_code=404, detail="upload file failed!")
        filename = urllib.parse.quote(file.filename, safe="")
        save_path = os.path.join(DATASET_BASE_PATH, filename)
        await save_content_to_local_disk(save_path, file)

        fileBytes = os.path.getsize(save_path)
        fileInfo = FileObject(
            id=f"file-{uuid.uuid4()}",
            object="file",
            bytes=fileBytes,
            created_at=int(time.time()),
            filename=filename,
            purpose="fine-tune",
        )

        return fileInfo

    def invoke(self, *args, **kwargs):
        pass

    def check_health(self) -> bool:
        """Checks the health of the component.

        Returns:
            bool: True if the component is healthy, False otherwise.
        """
        return True
