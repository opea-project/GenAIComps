# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import asyncio
from comps import OpeaDataprepInterface, CustomLogger, OpeaComponentController


logger = CustomLogger("OpeaDataprepController")
logflag = os.getenv("LOGFLAG", False)


class OpeaDataprepController(OpeaComponentController):
    def __init__(self):
        super().__init__()

    def invoke(self, *args, **kwargs):
        pass

    async def ingest_files(self, *args, **kwargs):
        return self.active_component.ingest_files(*args, **kwargs)
    
    async def get_files(self, *args, **kwargs):
        return self.active_component.get_files(*args, **kwargs)
    
    async def delete_files(self, *args, **kwargs):
        return self.active_component.delete_files(*args, **kwargs)

