from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel


class FineTuningJobsRequest(BaseModel):
    training_file: str
    model: str


class Hyperparameters(BaseModel):
    n_epochs: int
    batch_size: int
    learning_rate_multiplier: float


class FineTuningJob(BaseModel):
    object: str = "fine_tuning.job"  # Set as constant
    id: str
    model: str
    created_at: int
    finished_at: int = None
    fine_tuned_model: str = None
    organization_id: str = None
    result_files: List[str] = None
    status: str
    validation_file: str = None
    training_file: str
    hyperparameters: Hyperparameters
    trained_tokens: int = None
    integrations: List[str] = []  # Empty list by default
    seed: int
    estimated_finish: int = 0  # Set default value to 0


class FineTuningJobList(BaseModel):
    object: str = "list"  # Set as constant
    data: List[FineTuningJob]
    has_more: bool


class FineTuningJobEvent(BaseModel):
    object: str = "fine_tuning.job.event"  # Set as constant
    id: str
    created_at: int
    level: str
    message: str
    data: None = None  # No data expected for this event type, set to None
    type: str = "message"  # Default event type is "message"
