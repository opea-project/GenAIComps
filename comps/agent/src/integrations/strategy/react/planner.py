# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent as create_react_langchain_agent
from langchain.memory import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import ChatHuggingFace
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from comps.cores.telemetry.opea_telemetry import opea_telemetry, tracer

from ...global_var import threads_global_kv
from ...storage.persistence_redis import RedisPersistence
from ...utils import filter_tools, has_multi_tool_inputs, tool_renderer
from ..base_agent import BaseAgent


class ReActAgentwithLangchain(BaseAgent):
    @opea_telemetry
    def __init__(self, args, with_memory=False, **kwargs):
        super().__init__(args, local_vars=globals(), **kwargs)
        from .prompt import hwchase17_react_prompt

        prompt = hwchase17_react_prompt
        if has_multi_tool_inputs(self.tools_descriptions):
            raise ValueError("Only supports single input tools when using strategy == react_langchain")
        else:
            agent_chain = create_react_langchain_agent(
                self.llm, self.tools_descriptions, prompt, tools_renderer=tool_renderer
            )
        self.app = AgentExecutor(
            agent=agent_chain, tools=self.tools_descriptions, verbose=True, handle_parsing_errors=True
        )
        self.memory = {}

        def get_session_history(session_id):
            if session_id in self.memory:
                return self.memory[session_id]
            else:
                mem = ChatMessageHistory()
                self.memory[session_id] = mem
                return mem

        if with_memory:
            self.app = RunnableWithMessageHistory(
                self.app,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                history_factory_config=[],
            )

    def prepare_initial_state(self, query):
        return {"input": query}

    @opea_telemetry
    async def stream_generator(self, query, config, thread_id=None):
        initial_state = self.prepare_initial_state(query)
        if thread_id is not None:
            config["configurable"] = {"session_id": thread_id}
        else:
            config["configurable"] = {"session_id": "0"}
        async for chunk in self.app.astream(initial_state, config=config):
            if thread_id is not None:
                with threads_global_kv as g_threads:
                    thread_inst, created_at, status = g_threads[thread_id]
                    if status == "try_cancel":
                        yield "[thread_completion_callback] signal to cancel! Changed status to ready"
                        print("[thread_completion_callback] signal to cancel! Changed status to ready")
                        g_threads[thread_id] = (thread_inst, created_at, "ready")
                        break
            if "actions" in chunk:
                for action in chunk["actions"]:
                    yield f"Calling Tool: `{action.tool}` with input `{action.tool_input}`\n\n"
            # Observation
            elif "steps" in chunk:
                for step in chunk["steps"]:
                    yield f"Tool Result: `{step.observation}`\n\n"
            # Final result
            elif "output" in chunk:
                yield f"data: {repr(chunk['output'])}\n\n"
            else:
                raise ValueError()
            print("---")
        yield "data: [DONE]\n\n"


class ReActAgentwithLanggraph(BaseAgent):
    @opea_telemetry
    def __init__(self, args, with_memory=False, **kwargs):
        super().__init__(args, local_vars=globals(), **kwargs)
        if kwargs.get("custom_prompt") is not None:
            print("***Custom prompt is provided.")
            REACT_SYS_MESSAGE = kwargs.get("custom_prompt").REACT_SYS_MESSAGE
        else:
            print("*** Using default prompt.")
            from .prompt import REACT_SYS_MESSAGE
        tools = self.tools_descriptions
        print("REACT_SYS_MESSAGE: ", REACT_SYS_MESSAGE)

        if with_memory:
            self.app = create_react_agent(self.llm, tools=tools, prompt=REACT_SYS_MESSAGE, checkpointer=MemorySaver())
        else:
            self.app = create_react_agent(self.llm, tools=tools, prompt=REACT_SYS_MESSAGE)

    def prepare_initial_state(self, query):
        return {"messages": [HumanMessage(content=query)]}

    @opea_telemetry
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

    @opea_telemetry
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


###############################################################################
# ReActAgentLlama:
# Only validated with with Llama3.1-70B-Instruct model served with TGI-gaudi
# support multiple tools
# does not rely on langchain bind_tools API
# since tgi and vllm still do not have very good support for tool calling like OpenAI

import json
from typing import Annotated, Dict, List, Optional, Sequence, TypedDict, Union

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode

from ...storage.persistence_memory import AgentPersistence, PersistenceConfig
from ...utils import setup_chat_model
from .utils import (
    assemble_history,
    assemble_memory,
    assemble_memory_from_store,
    convert_aimessage_to_chat_completion,
    convert_json_to_tool_call,
    convert_think_to_chat_completion,
    save_state_to_store,
)


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_choice: Optional[List[str]] = None
    is_last_step: IsLastStep


class ReActAgentNodeLlama:
    """Do planning and reasoning and generate tool calls.

    A workaround for open-source llm served by TGI-gaudi.
    """

    @opea_telemetry
    def __init__(self, tools, args, store=None, **kwargs):
        from .utils import ReActLlamaOutputParser

        if kwargs.get("custom_prompt") is not None:
            print("***Custom prompt is provided.")
            REACT_AGENT_LLAMA_PROMPT = kwargs.get("custom_prompt").REACT_AGENT_LLAMA_PROMPT
        else:
            print("*** Using default prompt.")
            from .prompt import REACT_AGENT_LLAMA_PROMPT

        print("***Prompt template:\n", REACT_AGENT_LLAMA_PROMPT)

        output_parser = ReActLlamaOutputParser()
        prompt = PromptTemplate(
            template=REACT_AGENT_LLAMA_PROMPT,
            input_variables=["input", "history", "tools"],
        )
        llm = setup_chat_model(args)
        self.tools = tools
        self.chain = prompt | llm
        self.output_parser = output_parser
        self.with_memory = args.with_memory
        self.memory_type = args.memory_type
        self.store = store

    @opea_telemetry
    def __llm_invoke__(self, query, history, tools_descriptions, thread_history):
        # invoke chain: raw output from llm
        response = self.chain.invoke(
            {"input": query, "history": history, "tools": tools_descriptions, "thread_history": thread_history}
        )
        return response

    @opea_telemetry
    def __call__(self, state, config):

        print("---CALL Agent LLM node---")
        messages = state["messages"]

        # assemble a prompt from messages
        if self.with_memory:
            if self.memory_type == "checkpointer":
                query, history, thread_history = assemble_memory(messages)
            elif self.memory_type == "store":
                # use thread_id, assistant_id to search memory from store
                print("@@@ Load memory from store....")
                query, history, thread_history = assemble_memory_from_store(config, self.store)  # TODO
            else:
                raise ValueError("Invalid memory type!")
        else:
            query = messages[0].content
            history = assemble_history(messages)
            thread_history = ""

        print("@@@ Turn History:\n", history)
        print("@@@ Thread history:\n", thread_history)

        tools_used = self.tools
        if state.get("tool_choice") is not None:
            tools_used = filter_tools(self.tools, state["tool_choice"])

        tools_descriptions = tool_renderer(tools_used)
        print("@@@ Tools description: ", tools_descriptions)

        # invoke chain: raw output from llm
        response = self.__llm_invoke__(query, history, tools_descriptions, thread_history)

        content = response.content

        # parse tool calls or answers from raw output: result is a list
        output = self.output_parser.parse(content)
        print("@@@ Output from chain: ", output)

        # convert output to tool calls
        tool_calls = []
        if output:
            for res in output:
                if "tool" in res:
                    tool_call = convert_json_to_tool_call(res)
                    tool_calls.append(tool_call)

            if tool_calls:
                response.tool_calls = tool_calls
            elif "answer" in output[0]:
                response.content = str(output[0]["answer"])

        return {"messages": [response]}


class ReActAgentLlama(BaseAgent):
    @opea_telemetry
    def __init__(self, args, **kwargs):
        super().__init__(args, local_vars=globals(), **kwargs)

        agent = ReActAgentNodeLlama(tools=self.tools_descriptions, args=args, store=self.store, **kwargs)
        tool_node = ToolNode(self.tools_descriptions)

        workflow = StateGraph(AgentState)

        # Define the nodes we will cycle between
        workflow.add_node("agent", agent)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            self.should_continue,
            # Finally we pass in a mapping.
            # The keys are strings, and the values are other nodes.
            # END is a special node marking that the graph should finish.
            # What will happen is we will call `should_continue`, and then the output of that
            # will be matched against the keys in this mapping.
            # Based on which one it matches, that node will then be called.
            {
                # If `tools`, then we call the tool node.
                "continue": "tools",
                # Otherwise we finish.
                "end": END,
            },
        )

        # We now add a normal edge from `tools` to `agent`.
        # This means that after `tools` is called, `agent` node is called next.
        workflow.add_edge("tools", "agent")

        if args.with_memory:
            if args.memory_type == "checkpointer":
                self.app = workflow.compile(checkpointer=self.checkpointer)
            elif args.memory_type == "store":
                self.app = workflow.compile(store=self.store)
            else:
                raise ValueError("Invalid memory type!")
        else:
            self.app = workflow.compile()

    # Define the function that determines whether to continue or not
    def should_continue(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    def prepare_initial_state(self, query):
        print("---Prepare initial state---")
        return {"messages": [HumanMessage(content=query)]}

    @opea_telemetry
    async def stream_generator(self, query, config, thread_id=None):
        initial_state = self.prepare_initial_state(query)
        if "tool_choice" in config:
            initial_state["tool_choice"] = config.pop("tool_choice")

        stream_mode = ["updates", "messages"]
        try:
            print("---Start running---")
            react = "<think>"
            async for event in self.app.astream(initial_state, config=config, stream_mode=stream_mode):
                print(event)
                event_type = event[0]
                data = event[1]

                if event_type == "updates":
                    for node_name, node_state in data.items():
                        if self.memory_type == "store":
                            save_state_to_store(node_state, config, self.store)
                        print(f"--- CALL {node_name} node ---\n")

                        for k, v in node_state.items():
                            if k == "messages" and v is not None:
                                print(f"------- {k}, {v} -------\n\n")
                                if node_name == "agent":
                                    # the message is displayed in the event_type="messages"
                                    if "messages" not in stream_mode:
                                        for each in v:
                                            result = convert_aimessage_to_chat_completion(each)
                                            yield f"data: {json.dumps(result)}\n\n"

                                    # if no tool call, display final answer
                                    if "messages" in stream_mode:
                                        for each in v:
                                            if not each.tool_calls:
                                                result = convert_aimessage_to_chat_completion(
                                                    each, stream=True, metadata={}
                                                )
                                                yield f"data: {json.dumps(result)}\n\n"
                                elif node_name == "tools":
                                    # for multi tools in 1 turn
                                    for each in v:
                                        full_content = each.content
                                        tool_name = each.name
                                        result = {"tool_name": tool_name, "tool_content": [full_content]}
                                        yield f"data: {json.dumps(result)}\n\n"
                                        if not full_content:
                                            continue
                elif event_type == "messages":
                    # only keep AIMessageChunk message
                    if not isinstance(data[0], AIMessageChunk):
                        continue

                    # convert to openai format
                    result = convert_aimessage_to_chat_completion(data[0], stream=True, metadata=data[1])
                    yield f"data: {json.dumps(result)}\n\n"

                    # for ui display thinking process
                    if react == "<think>":
                        react = ""
                        result = convert_think_to_chat_completion("<think>")
                        yield f"data: {json.dumps(result)}\n\n"
                    if data[0].response_metadata.get("finish_reason") == "stop":
                        react = "<think>"
                        result = convert_think_to_chat_completion("</think>")
                        yield f"data: {json.dumps(result)}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield str(e)

    @opea_telemetry
    async def non_streaming_run(self, query, config):
        # for use as worker agent (tool of supervisor agent)
        # only used in chatcompletion api
        # chat completion api only supports checkpointer memory
        initial_state = self.prepare_initial_state(query)
        if "tool_choice" in config:
            initial_state["tool_choice"] = config.pop("tool_choice")
        try:
            async for s in self.app.astream(initial_state, config=config, stream_mode="values"):
                message = s["messages"][-1]
                message.pretty_print()
            last_message = s["messages"][-1]
            print("******Response: ", last_message.content)
            return last_message.content
        except Exception as e:
            return str(e)
