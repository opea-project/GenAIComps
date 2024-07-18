# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

env_config = []

if os.environ.get("port") is not None:
    env_config += ["--port", os.environ["port"]]

if os.environ.get("AGENT_NAME") is not None:
    env_config += ["--agent_name", os.environ["AGENT_NAME"]]

if os.environ.get("strategy") is not None:
    env_config += ["--strategy", os.environ["strategy"]]

if os.environ.get("llm_endpoint_url") is not None:
    env_config += ["--llm_endpoint_url", os.environ["llm_endpoint_url"]]

if os.environ.get("llm_engine") is not None:
    env_config += ["--llm_engine", os.environ["llm_engine"]]

if os.environ.get("model") is not None:
    env_config += ["--model", os.environ["model"]]

if os.environ.get("recursion_limit") is not None:
    env_config += ["--recursion_limit", os.environ["recursion_limit"]]

if os.environ.get("require_human_feedback") is not None:
    if os.environ["require_human_feedback"].lower() == "true":
        env_config += ["--require_human_feedback"]

if os.environ.get("debug") is not None:
    if os.environ["debug"].lower() == "true":
        env_config += ["--debug"]

if os.environ.get("role_description") is not None:
    env_config += ["--role_description", "'" + os.environ["role_description"] + "'"]

if os.environ.get("tools") is not None:
    env_config += ["--tools", os.environ["tools"]]

if os.environ.get("streaming") is not None:
    env_config += ["--streaming", os.environ["streaming"]]
