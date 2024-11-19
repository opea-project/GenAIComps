# Copyright (c) 2015-2024 MinIO, Inc.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import List, Optional
from urllib.parse import unquote

from pydantic import BaseModel, Field, validator


class UserIdentity(BaseModel):
    principalId: str


class RequestParameters(BaseModel):
    principalId: str
    region: str
    sourceIPAddress: str


class ResponseElements(BaseModel):
    x_amz_id_2: str = Field(..., alias="x-amz-id-2")
    x_amz_request_id: str = Field(..., alias="x-amz-request-id")
    x_minio_deployment_id: str = Field(..., alias="x-minio-deployment-id")
    x_minio_origin_endpoint: str = Field(..., alias="x-minio-origin-endpoint")


class BucketOwnerIdentity(BaseModel):
    principalId: str


class Bucket(BaseModel):
    name: str
    ownerIdentity: BucketOwnerIdentity
    arn: str


class ObjectUserMetadata(BaseModel):
    content_type: str = Field(..., alias="content-type")
    chunk_overlap: Optional[int] = Field(100, alias="X-Amz-Meta-Chunk_overlap")
    chunk_size: Optional[int] = Field(1500, alias="X-Amz-Meta-Chunk_size")
    process_table: Optional[bool] = Field(False, alias="X-Amz-Meta-Process_table")
    table_strategy: Optional[str] = Field("fast", alias="X-Amz-Meta-Table_strategy")

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class S3Object(BaseModel):
    key: str
    size: Optional[int] = None
    eTag: Optional[str] = None
    contentType: Optional[str] = None
    userMetadata: Optional[ObjectUserMetadata] = None
    sequencer: str

    @validator("key")
    def decode_key(cls, v):
        """Decode URL-encoded key."""
        return unquote(v)


class MinIOS3(BaseModel):
    s3SchemaVersion: str
    configurationId: str
    bucket: Bucket
    object: S3Object


class MinIOSource(BaseModel):
    host: str
    port: str
    userAgent: str


class MinIORecord(BaseModel):
    eventVersion: str
    eventSource: str
    awsRegion: str
    eventTime: datetime
    eventName: str
    userIdentity: UserIdentity
    requestParameters: RequestParameters
    responseElements: ResponseElements
    s3: MinIOS3 = Field(..., alias="s3")
    source: MinIOSource = Field(..., alias="source")


class MinioEventNotification(BaseModel):
    EventName: str
    Key: str
    Records: List[MinIORecord] = Field(..., alias="Records")

    class Config:
        from_attributes = True
        populate_by_name = True
