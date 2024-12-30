# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os

from comps import CustomLogger, OpeaComponentController

logger = CustomLogger("opea_vectorstores_controller")
logflag = os.getenv("LOGFLAG", False)


class OpeaVectorstoresController(OpeaComponentController):
    def __init__(self):
        super().__init__()

    def invoke(self, *args, **kwargs):
        pass

    def is_empty(self):
        if logflag:
            logger.info("[ vectorstores controller ] is empty")
        return self.active_component.is_empty()

    async def ingest_chunks(self, *args, **kwargs):
        if logflag:
            logger.info("[ vectorstores controller ] ingest chunks")
        return await self.active_component.ingest_chunks(*args, **kwargs)

    async def check_file_existance(self, *args, **kwargs):
        if logflag:
            logger.info("[ vectorstores controller ] check file existence")
        return await self.active_component.check_file_existance(*args, **kwargs)

    async def get_file_list(self, *args, **kwargs):
        if logflag:
            logger.info("[ vectorstores controller ] get file list")
        return await self.active_component.get_file_list(*args, **kwargs)

    async def get_file_content(self, *args, **kwargs):
        if logflag:
            logger.info("[ vectorstores controller ] get file content")
        return await self.active_component.get_file_content(*args, **kwargs)

    async def delete_all_files(self, *args, **kwargs):
        if logflag:
            logger.info("[ vectorstores controller ] delete all files")
        return await self.active_component.delete_all_files(*args, **kwargs)

    async def delete_single_file(self, *args, **kwargs):
        if logflag:
            logger.info("[ vectorstores controller ] delete single file")
        return await self.active_component.delete_single_file(*args, **kwargs)

    async def similarity_search(self, *args, **kwargs):
        if logflag:
            logger.info("[ vectorstores controller ] similarity search")
        return await self.active_component.similarity_search(*args, **kwargs)
