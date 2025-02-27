# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver

from ..storage.persistence_redis import RedisPersistence
from ..tools import get_tools_descriptions
from ..utils import adapt_custom_prompt, setup_chat_model


class BaseAgent:
    def __init__(self, args, local_vars=None, **kwargs) -> None:
        self.llm = setup_chat_model(args)
        self.tools_descriptions = get_tools_descriptions(args.tools)
        self.app = None
        self.id = f"assistant_{self.__class__.__name__}_{uuid4()}"
        self.args = args
        adapt_custom_prompt(local_vars, kwargs.get("custom_prompt"))
        print("Registered tools: ", self.tools_descriptions)

        if args.with_memory:
            if args.memory_type == "checkpointer":
                self.memory_type = "checkpointer"
                self.checkpointer = MemorySaver()
                self.store = None
            elif args.memory_type == "store":
                # print("Using Redis as store: ", args.store_config.redis_uri)
                self.store = RedisPersistence(args.store_config.redis_uri)
                self.memory_type = "store"
            else:
                raise ValueError("Invalid memory type!")
        else:
            self.store = None
            self.checkpointer = None

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

    def prepare_initial_state(self, query):
        raise NotImplementedError

    async def stream_generator(self, query, config):
        initial_state = self.prepare_initial_state(query)
        try:
            async for event in self.app.astream(initial_state, config=config):
                for node_name, node_state in event.items():
                    yield f"--- CALL {node_name} ---\n"
                    for k, v in node_state.items():
                        if v is not None:
                            yield f"{k}: {v}\n"

                yield f"data: {repr(event)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield str(e)

    async def non_streaming_run(self, query, config):
        initial_state = self.prepare_initial_state(query)
        print("@@@ Initial State: ", initial_state)
        try:
            async for s in self.app.astream(initial_state, config=config, stream_mode="values"):
                message = s["messages"][-1]
                message.pretty_print()

            last_message = s["messages"][-1]
            print("******Response: ", last_message.content)
            return last_message.content
        except Exception as e:
            return str(e)
