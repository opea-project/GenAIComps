# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
import operator
import os
import warnings
from enum import Enum
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ...global_var import threads_global_kv
from ...utils import has_multi_tool_inputs, tool_renderer
from ..base_agent import BaseAgent
from .prompt import DATA_RETRIEVER_SYS_PROMPT, REASONING_PROMPT, SCHEDULER_SYS_PROMPT, STATUS_CHECKER_SYS_PROMPT
from .utils import assemble_history


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    workflow_status: str
    workflow_output_data: str


class WorkflowSchedulerNode:

    def __init__(self, llm, tools):
        prompt = PromptTemplate(
            template=SCHEDULER_SYS_PROMPT,
            input_variables=["input", "history"],
        )
        self.chain = prompt | llm.bind_tools(tools) | (lambda x: x.tool_calls[0]["args"]) | workflow_scheduler

    def __call__(self, state):
        print("---CALL WorkflowScheduler node---")
        messages = state["messages"]

        response = self.chain.invoke(messages)

        return {"messages": [HumanMessage(content=response)]}


class WorkflowStatusCheckerNode:

    def __init__(self, llm, tools):

        prompt = PromptTemplate(
            template=STATUS_CHECKER_SYS_PROMPT,
            input_variables=["input", "history"],
        )
        self.chain = prompt | llm.bind_tools(tools) | (lambda x: x.tool_calls[0]["args"]) | status_checker

    def __call__(self, state):
        print("---CALL WorkflowStatusChecker node---")
        messages = state["messages"]

        query = messages[0].content
        history = assemble_history(messages)

        response = self.chain.invoke({"input": query, "history": history})

        return {"messages": [AIMessage(content=response["message"])], "workflow_status": response["status"]}


class WorkflowDataRetrieverNode:

    def __init__(self, llm, tools):
        prompt = PromptTemplate(
            template=DATA_RETRIEVER_SYS_PROMPT,
            input_variables=["input", "history"],
        )
        self.chain = prompt | llm.bind_tools(tools) | (lambda x: x.tool_calls[0]["args"]) | workflow_data_retriever

    def __call__(self, state):
        print("---CALL WorkflowDataRetriever node---")
        messages = state["messages"]

        query = messages[0].content
        history = assemble_history(messages)
        # print("@@@ History: ", history)

        response = self.chain.invoke({"input": query, "history": history})

        return {"messages": [AIMessage(content=response)], "workflow_output_data": response}


def reasoning_agent(state, llm):
    reasoning_prompt = PromptTemplate(
        template=REASONING_PROMPT,
        input_variables=["input", "tool_output"],
    )
    chain = reasoning_prompt | llm

    messages = state["messages"]
    query = messages[0].content
    tool_output = state["workflow_output_data"]

    response = chain.invoke({"input": query, "tool_output": tool_output})
    return {"messages": [HumanMessage(content=response.content)]}


def should_retry(state):
    MAX_RETRY = 3
    num_retry = 0
    instruction = "Workflow execution is still in progress."
    exceed_retries_message = "Total number of retries exceeded and workflow is still in progress. Exiting graph."

    for m in state["messages"]:
        if instruction in m.content:
            num_retry += 1

    print("**********Num retry: ", num_retry)

    if state["workflow_status"] == "failed":
        print("Workflow execution failed. Exiting graph")
        return "end"
    elif (num_retry < MAX_RETRY) and (state["workflow_status"] == "finished"):
        return True
    elif (num_retry < MAX_RETRY) and (not state["workflow_status"] == "finished"):
        return False
    else:
        print(exceed_retries_message)
        return "end"


class WorkflowExecutorAgentWithLangGraph(BaseAgent):
    def __init__(self, args, with_memory=False):
        super().__init__(args)

        workflow_scheduler_node = WorkflowSchedulerNode(self.llm_endpoint, self.tools_descriptions)
        status_checker_node = WorkflowStatusCheckerNode(self.llm_endpoint, self.tools_descriptions)
        workflow_retriever_node = WorkflowDataRetrieverNode(self.llm_endpoint, self.tools_descriptions)

        reasoning_node = functools.partial(reasoning_agent, llm=self.llm_endpoint)

        workflow = StateGraph(AgentState)
        workflow.add_node("workflow_scheduler", workflow_scheduler_node)
        workflow.add_node("workflow_status_checker", status_checker_node)
        workflow.add_node("workflow_data_retriever", workflow_retriever_node)
        workflow.add_node("reasoning_agent", reasoning_node)

        workflow.add_edge(START, "workflow_scheduler")
        workflow.add_edge("workflow_scheduler", "workflow_status_checker")

        workflow.add_conditional_edges(
            "workflow_status_checker",
            should_retry,
            {True: "workflow_data_retriever", False: "workflow_status_checker", "end": END},
        )

        workflow.add_edge("workflow_data_retriever", "reasoning_agent")
        workflow.add_edge("reasoning_agent", END)
        if with_memory:
            self.app = workflow.compile(checkpointer=MemorySaver())
        else:
            self.app = workflow.compile()

    def prepare_initial_state(self, query):
        return {"messages": [("user", query)]}

    async def stream_generator(self, query, config, thread_id=None):
        pass
