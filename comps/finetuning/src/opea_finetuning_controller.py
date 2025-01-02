# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from comps import CustomLogger, OpeaComponentController

logger = CustomLogger("opea_finetuning_controller")


class OpeaFinetuningController(OpeaComponentController):
    def __init__(self):
        super().__init__()

    def invoke(self, *args, **kwargs):
        pass

    def create_finetuning_jobs(self, *args, **kwargs):
        return self.active_component.create_finetuning_jobs(*args, **kwargs)

    def cancel_finetuning_job(self, *args, **kwargs):
        return self.active_component.cancel_finetuning_job(*args, **kwargs)

    def list_finetuning_checkpoints(self, *args, **kwargs):
        return self.active_component.list_finetuning_checkpoints(*args, **kwargs)

    def list_finetuning_jobs(self, *args, **kwargs):
        return self.active_component.list_finetuning_jobs(*args, **kwargs)

    def retrieve_finetuning_job(self, *args, **kwargs):
        return self.active_component.retrieve_finetuning_job(*args, **kwargs)

    async def upload_training_files(self, *args, **kwargs):
        return await self.active_component.upload_training_files(*args, **kwargs)
