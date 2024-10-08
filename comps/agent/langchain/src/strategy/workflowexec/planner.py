# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import json
from typing import Annotated, Sequence, TypedDict, Dict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain.pydantic_v1 import BaseModel, Field

from ...global_var import threads_global_kv
from ...utils import has_multi_tool_inputs, tool_renderer
from ..base_agent import BaseAgent
from .prompt import DATA_RETRIEVER_SYS_PROMPT, REASONING_PROMPT, SCHEDULER_SYS_PROMPT, STATUS_CHECKER_SYS_PROMPT
from .utils import assemble_history, prepare_tool_call

MAX_RETRY = 50
instruction = "Workflow execution is still in progress."
exceed_retries_message = "Total number of retries exceeded and workflow is still in progress. Exiting graph."

class WorkflowParams(BaseModel):
    workflow_id: int = Field(description="Workflow id")
    params: Dict[str, str]= Field(description="Workflow paramaters. Dictionary keys can have whitespace")

class WorkflowKey(BaseModel):
    workflow_key: str = Field(description="Workflow key")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sender: str
    workflow_status: str

class WorkflowScheduler:
    """Invokes llm to generate a workflow_scheduler tool call based on the current state. The llm extracts the params and workflow_id from the user query.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the response appended to messages
    """

    def __init__(self, llm):
        prompt = PromptTemplate(
            template=SCHEDULER_SYS_PROMPT,
            input_variables=["input", "history"],
        )

        class workflow_scheduler(WorkflowParams):
            """Used to start the workflow with a specified id."""

        self.chain =  prompt | llm.bind_tools([workflow_scheduler])

    def __call__(self, state):
        print("---CALL WorkflowScheduler node---")
        messages = state["messages"]

        response = self.chain.invoke(messages)

        return prepare_tool_call(response, self.name)

class WorkflowStatusChecker:
    """Invokes llm to generate a workflow_status_checker tool call based on the current state. The llm extracts workflow_key from the conversation history.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the response appended to messages
    """

    def __init__(self, llm):

        prompt = PromptTemplate(
            template=STATUS_CHECKER_SYS_PROMPT,
            input_variables=["input", "history"],
        )

        class workflow_status_checker(WorkflowKey):
            """Used to check the execution status of the workflow."""

            workflow_key: str = Field(description="Workflow key")

        self.chain =  prompt | llm.bind_tools([workflow_status_checker])

    def __call__(self, state):
        print("---CALL WorkflowStatusChecker node---")
        messages = state["messages"]

        query = messages[0].content
        history = assemble_history(messages)
        print("@@@ History: ", history)

        response = self.chain.invoke({"input": query, "history": history})

        return prepare_tool_call(response, self.name)

class WorkflowDataRetriever:
    """Invokes llm to generate a workflow_data_retriever tool call based on the current state. The llm extracts workflow_key from the conversation history.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the response appended to messages
    """

    def __init__(self, llm):
        prompt = PromptTemplate(
            template=DATA_RETRIEVER_SYS_PROMPT,
            input_variables=["input", "history"],
        )

        class workflow_data_retriever(WorkflowKey):
            """Used to retrieve workflow output data."""

            workflow_key: str = Field(description="Workflow key")

        self.chain =  prompt | llm.bind_tools([workflow_data_retriever])

    def __call__(self, state):
        print("---CALL WorkflowDataRetriever node---")
        messages = state["messages"]

        # assemble a prompt from messages
        query = messages[0].content
        history = assemble_history(messages)
        print("@@@ History: ", history)

        response = self.chain.invoke({"input": query, "history": history})

        return prepare_tool_call(response, self.name)

class ReasoningNode:
    """Invokes llm to answer the user's orginal question using the workflow output data.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the response appended to messages
    """

    def __init__(self, llm):
        reasoning_prompt = PromptTemplate(
            template=REASONING_PROMPT,
            input_variables=["input", "tool_output"],
        )
        self.chain = reasoning_prompt | llm

    def __call__(self, state):
        messages = state["messages"]
        query = messages[0].content
        tool_output = messages[-1].content

        response = self.chain.invoke({"input": query, "tool_output": tool_output})
        return {"messages": [AIMessage(content=response.content)]}

class ToolChainNode:
    """Executes tool calls based on the current state. Returns the tool result in the form of AIMessage.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the response appended to messages
    """

    def __init__(self, tools):
        self.chain = ToolNode(tools)

    def __call__(self, state):
        messages = state["messages"]
        response = self.chain.invoke({"messages": messages})

        sender = state["sender"]

        if sender == "workflow_status_checker":
            response = json.loads(response["messages"][0].content)
            return {"messages": [AIMessage(content=response["message"])], "workflow_status": response["status"]}

        else:
            return {"messages": [AIMessage(content=response["messages"][0].content)]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    else:
        return "end"

def should_retry(state):
    num_retry = 0
    
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
        time.sleep(100)   # interval bewteen each status checking retry
        return False
    else:
        print(exceed_retries_message)
        return "end"

class WorkflowExecutorAgentWithLangGraph(BaseAgent):
    def __init__(self, args, with_memory=False):
        super().__init__(args)

        workflow_scheduler_node = WorkflowScheduler(self.llm_endpoint, "workflow_scheduler")
        status_checker_node = WorkflowStatusChecker(self.llm_endpoint, "workflow_status_checker")
        workflow_retriever_node = WorkflowDataRetriever(self.llm_endpoint, "workflow_data_retriever")
        reasoning_node = ReasoningNode(self.llm_endpoint)
        tool_node = ToolChainNode(self.tools_descriptions)

        workflow = StateGraph(AgentState)
        workflow.add_node("workflow_scheduler", workflow_scheduler_node)
        workflow.add_node("workflow_status_checker", status_checker_node)
        workflow.add_node("workflow_data_retriever", workflow_retriever_node)
        workflow.add_node("reasoning_agent", reasoning_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("status_checker_tool", tool_node)

        workflow.add_edge(START, "workflow_scheduler")

        workflow.add_conditional_edges(
            "workflow_scheduler",
            should_continue,
            {
                "call_tool": "tools",
                "end": END
            },
        )
        workflow.add_conditional_edges(
            "workflow_status_checker",
            should_continue,
            {
                "call_tool": "status_checker_tool",
                "end": END
            },
        )
        workflow.add_conditional_edges(
            "workflow_data_retriever",
            should_continue,
            {
                "call_tool": "tools",
                "end": END
            },
        )

        workflow.add_conditional_edges(
            "tools",
            lambda x: x["sender"],
            {
                "workflow_scheduler": "workflow_status_checker",
                "workflow_data_retriever": "reasoning_agent"
            },
        )

        workflow.add_conditional_edges(
            "status_checker_tool",
            should_retry,
            {
                True: "workflow_data_retriever",
                False: "workflow_status_checker",
                "end": END
            },
        )

        workflow.add_edge("reasoning_agent", END)

        if with_memory:
            self.app = workflow.compile(checkpointer=MemorySaver())
        else:
            self.app = workflow.compile()

    def prepare_initial_state(self, query):
        return {"messages": [HumanMessage(content=query)]}

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
        try:
            async for s in self.app.astream(initial_state, config=config, stream_mode="values"):
                message = s["messages"][-1]
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()

            last_message = s["messages"][-1]
            print("******Response: ", last_message.content)
            return last_message.content
        except Exception as e:
            return str(e)