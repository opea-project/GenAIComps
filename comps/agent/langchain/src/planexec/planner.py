from json import JSONDecodeError
from typing import Any, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_partial_json
from langchain_core.outputs import Generation
from .prompt import PLANNER_PROMPT_v3, REPLANNER, REWRITER_v2, format_template


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )
class Response(BaseModel):
    """Response to user."""

    response: str

TEXT_BREAKER_DICT = {
    "mistralai/Mixtral-8x7B-Instruct-v0.1":"[/INST]",
    "alpindale/WizardLM-2-8x22B":"ASSISTANT: ",
    "microsoft/Phi-3-mini-4k-instruct":"\n<|assistant|>",
    "microsoft/Phi-3-mini-128k-instruct":"\n<|assistant|>",
}

class JsonStrOutputParser(JsonOutputParser):
    def parse_result(self, result: List[Generation], *, partial: bool = False) -> Any:
        text = result[0].text
        text = text.strip()
        if partial:
            try:
                return parse_partial_json(text)
            except JSONDecodeError:
                try:
                    return eval(text)
                except:
                    return None
        else:
            try:
                return parse_partial_json(text)
            except JSONDecodeError as e:
                try:
                    return eval(text)
                except:
                    msg = f"Invalid json output: {text}"
                    raise OutputParserException(msg, llm_output=text) from e
        

def tool_renderer(tools):
    tool_strings = []
    for tool in tools:
        description = f"{tool.name} - {tool.description}"
        
        arg_schema = []
        for k, tool_dict in tool.args.items():
            k_type = tool_dict['type'] if 'type' in tool_dict else ""
            k_desc = tool_dict['description'] if 'description' in tool_dict else ""
            arg_schema.append(f"@ {k} : {k_type}, {k_desc}")
        
        tool_strings.append(f"{description}, args: {arg_schema}")
    return "\n".join(tool_strings)


def create_planner(args, llm, tools, planner_type):
    tool_descriptions = tool_renderer(tools)

    if planner_type == "initial_plan":
        prompt = PromptTemplate(
            template=format_template(args,PLANNER_PROMPT_v3),
            input_variables=["objective", "date"], # latest langchain removed input_variables arg
            partial_variables={"tools": tool_descriptions}
            )
    elif planner_type == "plan_rewriter":
        prompt = PromptTemplate.from_template(
            template = format_template(args,REWRITER_v2),
            partial_variables={"tools": tool_descriptions})
    elif planner_type == "replanner":
        prompt = PromptTemplate.from_template(
            template = format_template(args,REPLANNER),
            partial_variables={"tools": tool_descriptions})
    
    parser = JsonStrOutputParser()
    chain = prompt | llm | parser

    return chain