# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from .storage.persistence_redis import RedisPersistence
from .utils import load_python_prompt

agent = None


async def instantiate_agent(args):
    global agent
    strategy = args.strategy
    with_memory = args.with_memory

    if agent is None:

        if args.custom_prompt is not None:
            print(f">>>>>> custom_prompt enabled, {args.custom_prompt}")
            custom_prompt = load_python_prompt(args.custom_prompt)
        else:
            custom_prompt = None

        if strategy == "react_langchain":
            from .strategy.react import ReActAgentwithLangchain

            agent = ReActAgentwithLangchain(args, with_memory, custom_prompt=custom_prompt)
        elif strategy == "react_langgraph":
            from .strategy.react import ReActAgentwithLanggraph

            agent = ReActAgentwithLanggraph(args, with_memory, custom_prompt=custom_prompt)
        elif strategy == "react_llama":
            print("Initializing ReAct Agent with LLAMA")
            from .strategy.react import ReActAgentLlama

            agent = ReActAgentLlama(args, custom_prompt=custom_prompt)
        elif strategy == "plan_execute":
            from .strategy.planexec import PlanExecuteAgentWithLangGraph

            agent = PlanExecuteAgentWithLangGraph(args, with_memory, custom_prompt=custom_prompt)

        elif strategy == "rag_agent" or strategy == "rag_agent_llama":
            print("Initializing RAG Agent")
            from .strategy.ragagent import RAGAgent

            agent = RAGAgent(args, with_memory, custom_prompt=custom_prompt)
        elif strategy == "sql_agent_llama":
            print("Initializing SQL Agent Llama")
            from .strategy.sqlagent import SQLAgentLlama

            agent = SQLAgentLlama(args, with_memory, custom_prompt=custom_prompt)
        elif strategy == "sql_agent":
            print("Initializing SQL Agent")
            from .strategy.sqlagent import SQLAgent

            agent = SQLAgent(args, with_memory, custom_prompt=custom_prompt)
        else:
            raise ValueError(f"Agent strategy: {strategy} not supported!")

        await agent.async_init()

    return agent
