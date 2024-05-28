from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import create_openai_functions_agent
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain.pydantic_v1 import BaseModel
from enum import Enum
from langchain.tools import StructuredTool
from typing import TypedDict, Annotated, Sequence, Union
import operator
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

import os

from fastapi.responses import StreamingResponse
from langchain_community.llms import HuggingFaceEndpoint
from langsmith import traceable
from langchain_community.chat_models.huggingface import ChatHuggingFace

from comps import GeneratedDoc, LLMParamsDoc, ServiceType, opea_microservices, register_microservice


# Set up the system template with a variable for context
code_writing_system_template = """
Generate working code based on the user's request.

Write the working code in Python language.

Before sharing, double check your work. I will tip you $100 if your code is perfect.

Do not explain your work, just share working code.
"""

ut_writing_system_template = """
Generate unit test code for the given code based on the user's request.

Write the UT code in Python language.

Before sharing, double check your work. I will tip you $100 if your code is perfect.

Do not explain your work, just share UT code.
"""

document_writing_system_template = """
Based on the user's request, generate a document to summarize the functionality of the codes
and explain how to use these codes.

Write the document in the format of MarkDown.

You can add bolding or highlighting to highlight the key information as needed.

Before sharing, double check your work. I will tip you $100 if your code is perfect.

Do not explain your work, just share the markdown content.
"""

review_writing_system_template = """
Based on the user's request, generate a review comments on how to refine these code.

Write the review comments with sample code as needed.

Before sharing, double check your work. I will tip you $100 if your code is perfect.

Do not explain your work, just share the review comments.
"""



code_writing_system_message_prompt = SystemMessagePromptTemplate.from_template(code_writing_system_template)
ut_writing_system_message_prompt = SystemMessagePromptTemplate.from_template(ut_writing_system_template)
document_writing_system_message_prompt = SystemMessagePromptTemplate.from_template(document_writing_system_template)
review_writing_system_message_prompt = SystemMessagePromptTemplate.from_template(review_writing_system_template)


# Set up the human template with a variable for the request
code_writing_human_template = """
{request}
"""
code_writing_human_message_prompt = HumanMessagePromptTemplate.from_template(code_writing_human_template)

ut_writing_human_template = """
{request}
"""
ut_writing_human_message_prompt = HumanMessagePromptTemplate.from_template(ut_writing_human_template)


document_writing_human_template = """
{request}
"""
document_writing_human_message_prompt = HumanMessagePromptTemplate.from_template(document_writing_human_template)


review_writing_human_template = """
{request}
"""
review_writing_human_message_prompt = HumanMessagePromptTemplate.from_template(review_writing_human_template)


llm_endpoint = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")
hf_endpoint = HuggingFaceEndpoint(
    endpoint_url=llm_endpoint,
    max_new_tokens=1024,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    streaming=False,
    timeout=600,
)

model = ChatHuggingFace(llm=hf_endpoint)

code_writing_prompt = ChatPromptTemplate.from_messages([code_writing_system_message_prompt,
                                                        code_writing_human_message_prompt])
output_parser = StrOutputParser()

code_writing_runnable = code_writing_prompt | model | output_parser


ut_writing_prompt = ChatPromptTemplate.from_messages([ut_writing_system_message_prompt,
                                                      ut_writing_human_message_prompt])
output_parser = StrOutputParser()

ut_writing_runnable = ut_writing_prompt | model | output_parser


document_writing_prompt = ChatPromptTemplate.from_messages([document_writing_system_message_prompt,
                                                           document_writing_human_message_prompt])
output_parser = StrOutputParser()

document_writing_runnable = document_writing_prompt | model | output_parser

review_writing_prompt = ChatPromptTemplate.from_messages([review_writing_system_message_prompt,
                                                          review_writing_human_message_prompt])
output_parser = StrOutputParser()

review_writing_runnable = review_writing_prompt | model | output_parser


tools = []
pseudo_tools_visible = [
    "Write Code",
    "Write Unit Test Code",
    "Add Document",
    "Review Code"
]
pseudo_tools_hidden = [
    "Store Request",
]

agent_tools = tools + pseudo_tools_visible + pseudo_tools_hidden
print(agent_tools)


# Set the agent options, which is FINISH plus all tools, with the exception of the hidden tools
agent_options = ["FINISH"] + agent_tools
agent_options = [item for item in agent_options if item not in pseudo_tools_hidden]
print(agent_options)

RouteOptions = Enum("RouteOptions", {option: option for option in agent_options})


class RouteInput(BaseModel):
    next: RouteOptions


def route(route: str) -> str:
    return route


router = StructuredTool.from_function(
    func=route,
    name="route",
    description="Select the next team member to use",
    args_schema=RouteInput,
    return_direct=False,
)

system_prompt_initial = """
You are a supervisor tasked with managing a development team consisting of the following members: {members}.

Given the following feature request from a user, respond with the worker to act next.

Each worker will perform a task and respond with their results and status. 

You typically follow this pattern:

1) Write code to solve the problem
2) Write the unit test code for the code you wrote
3) Add the document file with introduction and usage example in the format of Markdown
4) Write the review comment of the code you wrote

After the review comment is written, respond with FINISH.
"""

# Get the prompt to use - you can modify this!
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt_initial),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
).partial(options=str(agent_options), members=", ".join(agent_tools))

# Choose the LLM that will drive the agent
stream_hf_endpoint = HuggingFaceEndpoint(
    endpoint_url=llm_endpoint,
    max_new_tokens=1024,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    streaming=True,
    timeout=600,
)

llm = ChatHuggingFace(llm=stream_hf_endpoint)

# Construct the OpenAI Functions agent
agent_runnable = create_openai_functions_agent(llm, [router], prompt)

class AgentState(TypedDict):
    # The list of previous messages in the conversation
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    # The user's original request
    original_request: str
    # The code sample being generated
    code: str
    # The unit test code being generated
    ut_code: str
    # The document being generated
    document: str
    # The review comment being generated
    review_comment: str
    # Track whether the code hsa been approved
    code_approved: bool


# Define the function that determines whether to continue or not
def should_continue(state):
    # If the agent outcome is an AgentFinish, then we return `exit` string
    # This will be used when setting up the graph to define the flow
    if isinstance(state["agent_outcome"], AgentFinish):
        return "end"
    # Otherwise, an AgentAction is returned
    # Here we return `continue` string
    # This will be used when setting up the graph to define the flow
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    # messages = state['messages']
    agent_outcome = agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}


def call_set_initial_state(state):
    messages = state["messages"]
    last_message = messages[-1]
    return {"original_request": last_message.content}


# Define the function to execute tools
def call_tool(state):
    # We construct an ToolInvocation from the function_call
    tool = state['agent_outcome'].tool_input['next']
    print("Running Tool: ", tool)
    
    if tool == "Write Code":
        print("============ writing code ============")
        res = code_writing_runnable.invoke({"request": state["original_request"]})
        new_message = AIMessage(content="You have code now")
        return {"code": res, "messages": [new_message]}
    elif tool == "Write Unit Test Code":
        print("============ writing ut code ============")
        res = ut_writing_runnable.invoke({"request": state["code"]})
        new_message = AIMessage(content="You have unit test code now")
        return {"ut_code": res, "messages": [new_message]}
    elif tool == "Add Document":
        print("============ adding document ============")
        res = document_writing_runnable.invoke({"request": state["code"]})
        print(f"---- res:{res}")
        new_message = AIMessage(content="You have document now")
        return {"document": res, "messages": [new_message]}
    elif tool == "Review Code":
        print("============ writing review ============")
        res = review_writing_runnable.invoke({"request": state["code"]})
        new_message = AIMessage(content="Code is reviewed")
        return {"code_approved": True, "review_comment": res, "messages": [new_message]}
    elif tool == "Save Code":
        print("Save Code")
    return

# Define a new graph
graph = StateGraph(AgentState)

# Define the two nodes we will cycle between
graph.add_node("agent", call_model)
graph.add_node("action", call_tool)
graph.add_node("initial_state", call_set_initial_state)

# Set the entrypoint
graph.set_entry_point("initial_state")

# Add a conditional edge
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

# Aadd the Normal Edges
graph.add_edge("action", "agent")
graph.add_edge("initial_state", "agent")

# Compile it
app = graph.compile()



@traceable(run_type="tool")
def post_process_text(text: str):
    if text == " ":
        return "data: @#$\n\n"
    if text == "\n":
        return "data: <br/>\n\n"
    if text.isspace():
        return None
    new_text = text.replace(" ", "@#$")
    return f"data: {new_text}\n\n"


@register_microservice(
    name="opea_service@agent_tgi",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=8000,
)
@traceable(run_type="llm")
def llm_generate(input: LLMParamsDoc):
    inputs = {"messages": [HumanMessage(content=input.query)]}
    async def stream_generator():
        for output in app.stream(inputs):
            # stream() yields dictionaries with output keyed by node name
            for key, value in output.items():
                yield f"data: {value}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


if __name__ == "__main__":

    feature_request = """
    Create a function that adds the two numbers together.
    """

    inputs = {"messages": [HumanMessage(content=feature_request)]}
    for output in app.stream(inputs):
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")

    opea_microservices["opea_service@agent_tgi"].start()