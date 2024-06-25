# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, TypedDict, Union

from langgraph.graph import END, StateGraph

from .tools import get_tools_descriptions
from .utils import setup_llm, tool_renderer


def instantiate_agent(args, strategy="react"):
    if strategy == "react":
        return ReActAgentwithLangchain(args)
    elif strategy == "planexec":
        return PlanExecuteAgentWithLangGraph(args)
    else:
        return BaseAgent(args)


class BaseAgent:
    def __init__(self, args) -> None:
        self.llm_endpoint = setup_llm(args)
        self.tools_descriptions = get_tools_descriptions(args.tools)
        self.app = None
        # print(self.tools_descriptions)

    def compile(self):
        pass

    def execute(self, state: dict):
        pass


class BaseAgentState(TypedDict):
    input: str
    date: str
    plan: Union[List[dict], str]
    plan_errors: List[str]
    past_steps: List[Tuple]  # Annotated[List[Tuple], operator.add]
    # num_replan: int
    # num_rewrite: int
    # new_plan: List[dict]
    response: str


class ReActAgentwithLangchain(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        from langchain import hub
        from langchain.agents import AgentExecutor, create_react_agent

        prompt = hub.pull("hwchase17/react")
        agent_chain = create_react_agent(self.llm_endpoint, self.tools_descriptions, prompt, tools_renderer=tool_renderer)
        self.app = AgentExecutor(
            agent=agent_chain, tools=self.tools_descriptions, verbose=True, handle_parsing_errors=True
        )


class PlanExecuteAgentWithLangGraph(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        from .planexec.planner import create_planner

        self.valid_tools, self.valid_args = self.get_valid_tools_and_args()

        self.planner = create_planner(args, self.llm_endpoint, self.tools_descriptions, planner_type="initial_plan")
        self.plan_rewriter = create_planner(
            args, self.llm_endpoint, self.tools_descriptions, planner_type="plan_rewriter"
        )
        self.replanner = create_planner(args, self.llm_endpoint, self.tools_descriptions, planner_type="replanner")

        self.args = args
        self.debug = args.debug

        self.app = self.compile_workflow()

        if self.app is None:
            raise ValueError("Failed to compile the app")

    def get_valid_tools_and_args(self):
        tools = self.tools_descriptions
        valid_tools = []
        valid_args = {}
        for i, tool in enumerate(tools):
            tool_name = tool.name
            args_names = list(tool.args.keys())
            valid_tools.append(tool_name)
            valid_args[valid_tools[i]] = args_names

        print("VALID_TOOLS: ", valid_tools)
        print("VALID_ARGS: ", valid_args)
        return valid_tools, valid_args

    def compile_workflow(self):

        workflow = StateGraph(BaseAgentState)

        # Add the plan node
        workflow.add_node("planner", self.plan_step)

        # add plan checker node
        workflow.add_node("plan_checker", self.check_plan)

        # add plan rewrite node
        workflow.add_node("rewriter", self.rewrite_plan)

        # Add the execution step
        workflow.add_node("executor", self.execute_step)

        # Add a replan node
        workflow.add_node("replan", self.replan_step)

        workflow.set_entry_point("planner")

        # From plan we go to plan check
        workflow.add_edge("planner", "plan_checker")

        # add conditional edge between plan checker and rewrite,
        # as well as plan checker and executor
        workflow.add_conditional_edges(
            "plan_checker",
            self.should_rewrite_plan,
            {
                True: "rewriter",
                False: "executor",
            },
        )

        # From rewriter, we go to plan checker
        workflow.add_edge("rewriter", "plan_checker")

        # From executor, we replan
        workflow.add_edge("executor", "replan")

        workflow.add_conditional_edges(
            "replan",
            # Next, we pass in the function that will determine which node is called next.
            self.should_end,
            {
                # If `tools`, then we call the tool node.
                True: END,
                False: "plan_checker",
            },
        )

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        return workflow.compile()

    ### Below are the functions that will be called by the workflow
    def execute_step(self, state: BaseAgentState):
        # task: a json dict in the form of
        # {'tool': 'get_info', 'input_schema': {'ticker': '<result_1>'}, 'output_key': 'result_2'}
        def execute_with_toolname(tool_list, tool_name, input):
            tool_inst = None
            for tool in tool_list:
                if tool.name == tool_name:
                    tool_inst = tool
                    break
            if tool_inst is None:
                raise ValueError(f"Tool {tool_name} not found in the tool list")
            return tool_inst.run(input)

        def execute_one_step(tool, input, past_steps):
            # tool: str
            # input: dict
            # past_steps: list of tuples [(task, output), ()]
            try:
                for k, v in input.items():
                    v = str(v)
                    if ("<" in v) and (">" in v):  # needs an output from a previous step
                        ref_v = v.split("<")[1].split(">")[0]
                        ref_v = "<" + ref_v + ">"
                        for task, output in past_steps:
                            if ref_v == "<" + task["output_key"] + ">":
                                input[k] = input[k].replace(ref_v, output)
                output = execute_with_toolname(self.tools_descriptions, tool, input)
            except Exception as e:
                output = "Error during {} execution: {}".format(tool, str(e))

            return output

        past_steps = []
        for task in state["plan"]:
            tool = task["tool"]
            input = task["input_schema"]

            print("Executing task: {}".format(task))
            output = execute_one_step(tool, input, past_steps)
            print("Tool {} output: {}".format(tool, output))

            past_steps.append((task, output))
        print("Execution trace: ", past_steps)
        return {"past_steps": past_steps}

    def plan_step(self, state: BaseAgentState):
        if not self.debug:
            output = self.planner.invoke({"objective": state["input"], "date": state["date"]})
            if self.args.llm_engine == "openai":
                output = output.content
            if "steps" in output:
                plan = output["steps"]
            return {"plan": plan}

        else:  # debug mode
            plan = state["plan"]["steps"]
            return {"plan": plan}  # planner output parser returns a dict {'steps':[]}

    def rewrite_plan(self, state: BaseAgentState):
        if not self.debug:
            output = self.plan_rewriter.invoke(
                {
                    "objective": state["input"],
                    "initial_plan": state["plan"],
                    "errors": state["plan_errors"],
                    "date": state["date"],
                }
            )
            if self.args.llm_engine == "openai":
                output = output.content
            print("New plan:\n", output)
            return {"plan": output}
        else:
            plan = state["new_plan"]  # output from rewriter
            # num_rewrite = state['num_rewrite'] + 1
            return {"plan": plan}

    def check_plan(self, state: BaseAgentState):
        # plan: [{}, {}]
        errors = []
        if type(state["plan"]) == str:  # exception has happened in planner
            errors.append(state["plan"])
            return {"plan_errors": errors}
        else:
            steps = state["plan"]
            for i, step in enumerate(steps):
                tool = step["tool"]
                # first check if using a valid tool
                if tool not in self.valid_tools:
                    errors.append("Called {} tool, which is not available.".format(tool))
                else:  # if using a valid tool, then check the args
                    # check if the args are all correct
                    input_schema = step["input_schema"]
                    for k, v in input_schema.items():
                        v = str(v)
                        if k not in self.valid_args[tool]:
                            errors.append("Invalid argument {} when calling {}".format(k, tool))
                        if ("<" in v) and (">" in v):  # needs an output from a previous step
                            ref_v = v.split("<")[1].split(">")[0]
                            past_steps = steps[:i]
                            past_outputs = []
                            for past_s in past_steps:
                                past_outputs.append(past_s["output_key"])
                            if ref_v not in past_outputs:
                                errors.append("Invalid reference {} when calling {}".format(ref_v, tool))
        return {"plan_errors": errors}

    def replan_step(self, state: BaseAgentState):
        if not self.debug:
            output = self.replanner.invoke(
                {
                    "objective": state["input"],
                    "plan": state["plan"],
                    "past_steps": state["past_steps"],
                    "date": state["date"],
                }
            )
            if self.args.llm_engine == "openai":
                output = output.content
            print(output)
            parsed_output = output
            if (type(parsed_output) == dict) and ("response" in parsed_output):
                return {"response": parsed_output["response"]}
            else:  # no response, then try to return a plan
                try:
                    return {"plan": parsed_output["steps"]}
                except Exception as e:  # parsing error
                    return {"plan": "Replanner parsing error: {}".format(str(e))}

        else:  # debug mode
            return {"response": "This is a response from the replanner."}

    def should_end(self, state: BaseAgentState):
        if "response" in state and state["response"]:
            return True
        else:
            return False

    def should_rewrite_plan(self, state: BaseAgentState):
        if "plan_errors" in state and len(state["plan_errors"]) > 0:
            return True
        else:
            return False
