# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict, Field


class AgentConfig(BaseModel):
    model: str = None
    instructions: str = None
    enable_session_persistence: bool = False
    with_memory: bool = False
    tools: str = None
    custom_prompt: str = None
