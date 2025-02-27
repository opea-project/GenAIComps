# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import List, Optional, Union

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from tqdm import tqdm

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType

from .utils import store_embeddings
from .utils.utils import process_all_videos, read_config
from .utils.vclip import vCLIP

VECTORDB_SERVICE_HOST_IP = os.getenv("VDMS_HOST", "0.0.0.0")
VECTORDB_SERVICE_PORT = os.getenv("VDMS_PORT", 55555)
collection_name = os.getenv("INDEX_NAME", "rag-vdms")

logger = CustomLogger("opea_dataprep_vdms_multimodal")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("OPEA_DATAPREP_MULTIMODALVDMS")
class OpeaMultimodalVdmsDataprep(OpeaComponent):
    """Dataprep component for Multimodal Redis ingestion and search services."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.DATAPREP.name.lower(), description, config)
        self.device = "cpu"
        self.upload_folder = "./uploaded_files/"

        # Perform health check
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaMultimodalVdmsDataprep health check failed.")

    def check_health(self) -> bool:
        """Checks the health of the Multimodal Redis service."""
        return True

    def invoke(self, *args, **kwargs):
        pass

    def setup_vclip_model(self, config, device="cpu"):
        model = vCLIP(config)
        return model

    def read_json(self, path):
        with open(path) as f:
            x = json.load(f)
        return x

    def store_into_vectordb(self, vs, metadata_file_path, dimensions):
        GMetadata = self.read_json(metadata_file_path)

        total_videos = len(GMetadata.keys())

        for idx, (video, data) in enumerate(tqdm(GMetadata.items())):
            metadata_list = []
            ids = []

            data["video"] = video
            video_name_list = [data["video_path"]]
            metadata_list = [data]
            if vs.selected_db == "vdms":
                vs.video_db.add_videos(
                    paths=video_name_list,
                    metadatas=metadata_list,
                    start_time=[data["timestamp"]],
                    clip_duration=[data["clip_duration"]],
                )
            else:
                logger.info(f"ERROR: selected_db {vs.selected_db} not supported. Supported:[vdms]")

        # clean up tmp_ folders containing frames (jpeg)
        for i in os.listdir():
            if i.startswith("tmp_"):
                logger.info("removing tmp_*")
                os.system("rm -r tmp_*")
                break

    def generate_video_id(self):
        """Generates a unique identifier for a video file."""
        return str(uuid.uuid4())

    def generate_embeddings(self, config, dimensions, vs):
        process_all_videos(config)
        global_metadata_file_path = os.path.join(config["meta_output_dir"], "metadata.json")
        logger.info(f"global metadata file available at {global_metadata_file_path}")
        self.store_into_vectordb(vs, global_metadata_file_path, dimensions)

    async def ingest_videos(self, files: List[UploadFile] = File(None)):
        """Ingest videos to VDMS."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config = read_config(os.path.join(current_dir, "./config/config.yaml"))
        meanclip_cfg = {
            "model_name": config["embeddings"]["vclip_model_name"],
            "num_frm": config["embeddings"]["vclip_num_frm"],
        }
        generate_frames = config["generate_frames"]
        path = config["videos"]
        meta_output_dir = config["meta_output_dir"]
        emb_path = config["embeddings"]["path"]
        host = VECTORDB_SERVICE_HOST_IP
        port = int(VECTORDB_SERVICE_PORT)
        selected_db = config["vector_db"]["choice_of_db"]
        vector_dimensions = config["embeddings"]["vector_dimensions"]
        logger.info(f"Parsing videos {path}.")

        # Saving videos
        if files:
            video_files = []
            for file in files:
                if os.path.splitext(file.filename)[1] == ".mp4":
                    video_files.append(file)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File {file.filename} is not an mp4 file. Please upload mp4 files only.",
                    )

            for video_file in video_files:
                video_id = self.generate_video_id()
                video_name = os.path.splitext(video_file.filename)[0]
                video_file_name = f"{video_name}_{video_id}.mp4"
                video_dir_name = os.path.splitext(video_file_name)[0]
                # Save video file in upload_directory
                with open(os.path.join(path, video_file_name), "wb") as f:
                    shutil.copyfileobj(video_file.file, f)

        # Creating DB
        logger.info(
            "Creating DB with video embedding and metadata support, \nIt may take few minutes to download and load all required models if you are running for first time."
        )
        logger.info("Connecting to {} at {}:{}".format(selected_db, host, port))

        # init meanclip model
        model = self.setup_vclip_model(meanclip_cfg, device="cpu")
        vs = store_embeddings.VideoVS(
            host, port, selected_db, model, collection_name, embedding_dimensions=vector_dimensions
        )
        logger.info("done creating DB, sleep 5s")
        await asyncio.sleep(5)

        self.generate_embeddings(config, vector_dimensions, vs)

        return {"message": "Videos ingested successfully"}

    async def get_videos(self):
        """Returns list of names of uploaded videos saved on the server."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config = read_config(os.path.join(current_dir, "./config/config.yaml"))
        if not Path(config["videos"]).exists():
            logger.info("No file uploaded, return empty list.")
            return []

        uploaded_videos = os.listdir(config["videos"])
        mp4_files = [file for file in uploaded_videos if file.endswith(".mp4")]
        return mp4_files

    async def get_one_file(self, filename: str):
        """Download the file from remote."""

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config = read_config(os.path.join(current_dir, "./config/config.yaml"))
        UPLOAD_DIR = config["videos"]
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            return FileResponse(path=file_path, filename=filename)
        else:
            return {"error": "File not found"}

    async def ingest_generate_transcripts(self, files: List[UploadFile] = File(None)):
        pass

    async def ingest_generate_caption(self, files: List[UploadFile] = File(None)):
        pass

    async def ingest_files(self, files: Optional[Union[UploadFile, List[UploadFile]]] = File(None)):
        pass

    async def get_files(self):
        pass

    async def delete_files(self, file_path):
        pass
