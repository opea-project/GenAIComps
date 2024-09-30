import os
import functools
import operator
from typing import Annotated, Sequence, TypedDict
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.pydantic_v1 import BaseModel

from ...global_var import threads_global_kv
from ...utils import has_multi_tool_inputs, tool_renderer
from ..base_agent import BaseAgent
from .prompt import (
    supervisor_prompt,
    agent_sys_prompt
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

class Next(str, Enum):
    workflow_scheduler = "workflow_scheduler"
    workflow_status_checker = "workflow_status_checker"
    workflow_data_retriever = "workflow_data_retriever"
    FINISH = "FINISH"

class routeResponse(BaseModel):
    next: Next

class SupervisorAgent():
    def __init__(self, llm_endpoint, members):
        self.llm_endpoint = llm_endpoint
        options = ["FINISH"] + members
        self.prompt = supervisor_prompt.partial(options=str(options), members=", ".join(members))

    def supervisor_agent(self, state):
        supervisor_chain = (
            self.prompt
            | self.llm_endpoint.with_structured_output(routeResponse, method="function_calling")
        )
        return supervisor_chain.invoke(state)

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}

class WorkflowExecutorAgentWithLangGraph(BaseAgent):
    def __init__(self, args, with_memory=False):
        super().__init__(args)

        members = [tool.name for tool in self.tools_descriptions]

        workflow_executor_agent = create_react_agent(self.llm_endpoint,
                                                    self.tools_descriptions,
                                                    messages_modifier=agent_sys_prompt+"You schedule the workflow using the parameters given."
                                                    )
        status_checker_agent = create_react_agent(self.llm_endpoint,
                                                self.tools_descriptions,
                                                messages_modifier=agent_sys_prompt + "You check if the workflow is finished. Return only the status."
                                                )
        workflow_retriever_agent = create_react_agent(self.llm_endpoint, 
                                                    self.tools_descriptions,
                                                    messages_modifier=agent_sys_prompt+ "You retrieve the workflow output data and answer the user's original question."
                                                    )

        workflow_executor_node = functools.partial(agent_node, agent=workflow_executor_agent, name="workflow_scheduler")
        status_checker_node = functools.partial(agent_node, agent=status_checker_agent, name="workflow_status_checker")
        workflow_retriever_node = functools.partial(agent_node, agent=workflow_retriever_agent, name="workflow_data_retriever")

        workflow = StateGraph(AgentState)
        workflow.add_node("workflow_scheduler", workflow_executor_node)
        workflow.add_node("workflow_status_checker", status_checker_node)
        workflow.add_node("workflow_data_retriever", workflow_retriever_node)
        workflow.add_node("supervisor", SupervisorAgent(self.llm_endpoint, members).supervisor_agent)

        for member in members:
            workflow.add_edge(member, "supervisor")
        conditional_map = {k: k for k in members}
        conditional_map["FINISH"] = END
        workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
        workflow.add_edge(START, "supervisor")

        if with_memory:
            self.app = workflow.compile(checkpointer=MemorySaver())
        else:
            self.app = workflow.compile()

    def prepare_initial_state(self, query):
        return {"messages": [("user", query)]}

    async def stream_generator(self, query, config, thread_id=None):
        pass
