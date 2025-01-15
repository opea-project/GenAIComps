# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os

from comps import CustomLogger, OpeaComponentLoader

logger = CustomLogger("opea_dataprep_loader")
logflag = os.getenv("LOGFLAG", False)


class OpeaDataprepLoader(OpeaComponentLoader):
    def __init__(self, component_name, **kwargs):
        super().__init__(component_name=component_name, **kwargs)

    def invoke(self, *args, **kwargs):
        pass

    async def ingest_files(self, *args, **kwargs):
        if logflag:
            logger.info("[ dataprep loader ] ingest files")
        return await self.component.ingest_files(*args, **kwargs)

    async def get_files(self, *args, **kwargs):
        if logflag:
            logger.info("[ dataprep loader ] get files")
        return await self.component.get_files(*args, **kwargs)

    async def delete_files(self, *args, **kwargs):
        if logflag:
            logger.info("[ dataprep loader ] delete files")
        return await self.component.delete_files(*args, **kwargs)


class OpeaDataprepMultiModalLoader(OpeaComponentLoader):
    def __init__(self, component_name, **kwargs):
        super().__init__(component_name=component_name, **kwargs)

    def invoke(self, *args, **kwargs):
        pass

    async def ingest_files(self, *args, **kwargs):
        if logflag:
            logger.info("[ dataprep loader ] ingest files")
        return await self.component.ingest_files(*args, **kwargs)

    async def ingest_videos(self, *args, **kwargs):
        if logflag:
            logger.info("[ dataprep loader ] ingest files")
        return await self.component.ingest_videos(*args, **kwargs)

    async def ingest_generate_transcripts(self, *args, **kwargs):
        if logflag:
            logger.info("[ dataprep loader ] ingest generate transcripts")
        return await self.component.ingest_generate_transcripts(*args, **kwargs)

    async def ingest_generate_captions(self, *args, **kwargs):
        if logflag:
            logger.info("[ dataprep loader ] ingest generate captions")
        return await self.component.ingest_generate_captions(*args, **kwargs)

    async def get_files(self, *args, **kwargs):
        if logflag:
            logger.info("[ dataprep loader ] get files")
        return await self.component.get_files(*args, **kwargs)

    async def get_one_file(self, *args, **kwargs):
        if logflag:
            logger.info("[ dataprep loader ] get one file")
        return await self.component.get_one_file(*args, **kwargs)

    async def get_videos(self, *args, **kwargs):
        if logflag:
            logger.info("[ dataprep loader ] get videos")
        return await self.component.get_videos(*args, **kwargs)

    async def delete_files(self, *args, **kwargs):
        if logflag:
            logger.info("[ dataprep loader ] delete files")
        return await self.component.delete_files(*args, **kwargs)
