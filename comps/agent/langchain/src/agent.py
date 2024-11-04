# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from comps.cores.proto.agents import AgentConfig 
from .utils import load_python_prompt


def instantiate_agent(args, strategy="react_langchain", with_memory=False):
    if args.custom_prompt is not None:
        print(f">>>>>> custom_prompt enabled, {args.custom_prompt}")
        custom_prompt = load_python_prompt(args.custom_prompt)
    else:
        custom_prompt = None

    agent_config = AgentConfig(
        model = args.llm_engine,
        with_memory = with_memory,
        custom_prompt = custom_prompt,
        tools = args.tools
        enable_session_persistence=False,
    )

    if strategy == "react_langchain":
        from .strategy.react import ReActAgentwithLangchain

        return ReActAgentwithLangchain(args, with_memory, custom_prompt=custom_prompt)
    elif strategy == "react_langgraph":
        from .strategy.react import ReActAgentwithLanggraph

        return ReActAgentwithLanggraph(args, with_memory, custom_prompt=custom_prompt)
    elif strategy == "react_llama":
        print("Initializing ReAct Agent with LLAMA")
        from .strategy.react import ReActAgentLlama

        return ReActAgentLlama(args, agent_config)
    elif strategy == "plan_execute":
        from .strategy.planexec import PlanExecuteAgentWithLangGraph

        return PlanExecuteAgentWithLangGraph(args, with_memory, custom_prompt=custom_prompt)

    elif strategy == "rag_agent" or strategy == "rag_agent_llama":
        print("Initializing RAG Agent")
        from .strategy.ragagent import RAGAgent

        return RAGAgent(args, with_memory, custom_prompt=custom_prompt)
    else:
        raise ValueError(f"Agent strategy: {strategy} not supported!")
