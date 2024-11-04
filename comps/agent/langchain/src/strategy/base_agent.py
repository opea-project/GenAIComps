# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

from ..tools import get_tools_descriptions
from ..utils import adapt_custom_prompt, setup_chat_model


class BaseAgent:
    def __init__(self, args, local_vars=None, agent_config=None, **kwargs) -> None:
        self.llm = setup_chat_model(args)
        self.tools_descriptions = get_tools_descriptions(args.tools)
        self.app = None
        self.memory = None
        self.id = f"assistant_{self.__class__.__name__}_{uuid4()}"
        self.args = args
        adapt_custom_prompt(local_vars, kwargs.get("custom_prompt"))
        print(self.tools_descriptions)

        self.storage = None
        if agent_config.enable_session_persistence:
            from llama_stack.providers.utils.kvstore import KVStoreConfig, kvstore_impl

            # need async
            # self.persistence_store = await kvstore_impl(self.config.persistence_store)
            self.persistence_store = await kvstore_impl(KVStoreConfig())

            await self.persistence_store.set(
                key=f"agent:{self.id}",
                value=agent_config.json(),
            )

            self.storage = AgentPersistence(self.id, self.persistence_store)

    @property
    def is_vllm(self):
        return self.args.llm_engine == "vllm"

    @property
    def is_tgi(self):
        return self.args.llm_engine == "tgi"

    @property
    def is_openai(self):
        return self.args.llm_engine == "openai"

    def compile(self):
        pass

    def execute(self, state: dict):
        pass

    def non_streaming_run(self, query, config):
        raise NotImplementedError

    async def create_session(self, name: str) -> str:
        return await self.storage.create_session(name)
