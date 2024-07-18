from typing import Annotated, Any, Sequence, TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from .prompt import rlm_rag_prompt
from ..base_agent import BaseAgent

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

class DocumentGrader:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """
    def __init__(self, llm_endpoint):
        class grade(BaseModel):
            """Binary score for relevance check."""

            binary_score: str = Field(description="Relevance score 'yes' or 'no'")
        llm_with_tool = llm_endpoint.with_structured_output(grade)
        
        # Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )
        self.chain = prompt | llm_with_tool

    def __call__(self, state) -> Literal["generate", "rewrite"]:
        print("---CALL DocumentGrader---")
        messages = state["messages"]
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content

        scored_result = self.chain.invoke({"question": question, "context": docs})

        score = scored_result.binary_score

        if score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "generate"

        else:
            print(f"---DECISION: DOCS NOT RELEVANT, score is {score}---")
            return "rewrite"
        
class RagAgent:
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    def __init__(self, llm_endpoint, tools):
        self.llm = llm_endpoint.bind_tools(tools)
        
    def __call__(self, state):
        print("---CALL RagAgent---")
        messages = state["messages"]
        
        response = self.llm.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

class Retriever:
    def __init__(self, tools_descriptions):
        self.tools = tools_descriptions
        
    def __call__(self, state):
        return ToolNode(self.tools)

class Rewriter:
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """
    def __init__(self, llm_endpoint):
        self.llm = llm_endpoint
        
    def __call__(self, state):
        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
            )
        ]

        response = self.llm.invoke(msg)
        return {"messages": [response]}

class TextGenerator:
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """
    def __init__(self, llm_endpoint):
        # Chain
        promt = rlm_rag_prompt
        self.rag_chain = prompt | llm_endpoint | StrOutputParser()
        
    def __call__(self, state):
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content

        # Run
        response = self.rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}

class RAGAgentwithLanggraph(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        
        # Define Nodes
        document_grader = DocumentGrader(self.llm_endpoint)
        rag_agent = RagAgent(self.llm_endpoint, self.tools_descriptions)
        retriever = Retriever(self.llm_endpoint)
        rewriter = Rewriter(self.llm_endpoint)
        text_generator = TextGenerator(self.llm_endpoint)
        
        # Define graph
        workflow = StateGraph(AgentState)

        # Define the nodes we will cycle between
        workflow.add_node("agent", rag_agent)
        workflow.add_node("retrieve", retriever)
        workflow.add_node("rewrite", rewriter)
        workflow.add_node("generate", text_generator)
        
        # connect as graph
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve", # if tools_condition return 'tools', then go to 'retrieve'
                END: END, # if tools_condition return 'END', then go to END
            },
        )
        workflow.add_conditional_edges(
            "retrieve",
            document_grader,
            {
                "generate": "generate", # if tools_condition return 'generate', then go to 'generate' node
                "rewrite": "rewrite", # if tools_condition return 'rewrite', then go to 'rewrite' node
            }
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")

        self.app = workflow.compile()   