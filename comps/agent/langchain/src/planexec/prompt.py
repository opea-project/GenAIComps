# from langchain import hub
# https://smith.langchain.com/hub/homanp/superagent
# hub.pull("homanp/superagent")

#https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
def format_template(args, template):
    model_type = args.model.split("/")[0]
    if model_type == "mistralai":
        template = '<s>[INST] '+template+' [/INST]'
    elif args.model == "alpindale/WizardLM-2-8x22B":
        template = "USER: "+template+"ASSISTANT: "
    elif "phi-3" in model_type.lower():
        template = "<|user|>\n"+template+"\n<|assistant|>"
    elif args.model == "cognitivecomputations/dolphin-2.9-llama3-70b":
        template = "<|im_start|>user\n"+template+"\n<|im_end|>"+"\n<|im_start|>assistant"
    elif model_type == "meta-llama":
        # ref: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
        template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"+template+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return template

##############################
PLANNER_BODY = """\
Each step must contain 3 elements: tool, input_schema and output_key. \
Each step can use ONLY ONE tool. Do NOT combine tool calls in one step. \
A step in the plan can receive the output from a previous step as input. \
Use <> to enclose an input when using an output from a previous step. \
Read tool descriptions carefully and only come up with steps that the tools can do. 

Tools: 
{tools}

You MUST strictly follow the output format below. You MUST NOT include any other text!
{{  
  "steps": [
    {{"tool": str, "input_schema": {{input_schema}}, "output_key": str}}
  ]
}}

Do NOT use f-strings or string operations in the steps. \
Do NOT include your thought or explanations in your output! \
Do NOT add comments to the steps!

Objective can be time sensitive. Pay attention to today's date when composing the plan. \
Today's date is: {date} \n
OBJECTIVE: {objective}\n
"""

INITIAL_PLANNER_HEADER = "Given the OBJECTIVE, create a step-by-step plan by only using the tools listed below.\n"

REWRITER_HEADER = """\
My objective was: {objective}\n
Original plan to achieve this objective was: {initial_plan}\n
But there are errors in the original plan: {errors}\n
Think out-of-box and create a new step-by-step plan by only using the tools listed below.\n
"""


PLANNER_PROMPT_v3 = """\
Given the OBJECTIVE, create a step-by-step plan by only using the tools listed below. \
Each step must contain 3 elements: tool, input_schema and output_key. \
Each step can use ONLY ONE tool. Do NOT combine tool calls in one step. \
A step in the plan can receive the output from a previous step as input. \
Use <> to enclose an input when using an output from a previous step. \
Read tool descriptions carefully and only come up with steps that the tools can do. 

Tools: 
{tools}

You MUST strictly follow the output format below. You MUST NOT include any other text!
{{  
  "steps": [
    {{"tool": str, "input_schema": {{input_schema}}, "output_key": str}}
  ]
}}

Do NOT use f-strings or string operations in the steps. \
Do NOT include your thought or explanations in your output! \
Do NOT add comments to the steps!

Objective can be time sensitive. Pay attention to today's date when composing the plan. \
Today's date is: {date} \n
OBJECTIVE: {objective}\n
"""

##############################
# Plan rewriter prompt
REWRITER_v3 = REWRITER_HEADER + PLANNER_BODY

REWRITER_v2 ="""\
My objective was: {objective}\n
Original plan to achieve this objective was: {initial_plan}\n
But there are errors in the original plan: {errors}\n
Create a new step-by-step plan to achieve my objective. Remember that you can only use the followings tools in the plan:
{tools} 

Each step in the plan is one dictionary with 3 keys: tool, input_schema and output_key. \
A step in the plan can receive the output from a previous step as input. Use <> to enclose an input when using an output from a previous step.\
Separate two function calls to two steps. Do NOT combine them into one step. \
You must conform to the input specs defined in the tool descriptions. 

Objective can be time sensitive. Pay attention to today's date when composing the plan. \
Today's date is: {date} \n
OBJECTIVE: {objective}\n

You must use the following format for the new plan.You MUST NOT include any other text.
{{  
  "steps": [
    {{"tool": str, "input_schema": {{input_schema}}, "output_key": str}}
  ]
}}
Think out-of-box and create a new plan to achieve the objective.
New Plan:
"""

##############################
# synthesize_replan prompt
# generate output given execution trace
# if answer can be generated, then output answer
# if cannot generate answer based on execution trace, then output new plan

REPLANNER ="""\
Your objective was this:
{objective}

Your original plan was this:
{plan}

You have done the following steps:
{past_steps}

If you can come up with an answer with info contained the past steps, then generate a response using the following JSON format.
{{
  "response": Your final answer here.
}}

Otherwise, think out-of-box and create a new plan to achieve the objective. Remember that you can only use the followings tools in the plan:
{tools}

You must use the following format for the new plan.
{{  
  "steps": [
    {{"tool": str, "input_schema": {{input_schema}}, "output_key": str}}
  ]
}}
You MUST output in the specified format. You MUST NOT include any other text. You MUST NOT repeat the old plan.
Objective can be time sensitive. Pay attention to today's date when composing the plan. \
Today's date is: {date}
Objective: {objective}

Output either the response or the new plan, not both.

YOUR OUTPUT:
"""


####################################################################################

# The planner prompt template
PLANNER_PROMPT_v0 = """For the given objective, create a plan to solve it with the utmost parallelizability. \
Each plan should comprise an action from the following {num_tools} types:
{tool_descriptions}
- Each action described above contains input/output types and description.
- You must strictly adhere to the input and output types for each action.
- The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.
- Each action in the plan should strictly be one of the above types. Follow the Python conventions for each action.
- Each action MUST have a unique ID, which is strictly increasing.
- Inputs for actions can either be constants or outputs from preceding actions. In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.
- Only use the provided action types. If a query cannot be addressed using these, invoke the join action for the next steps.
- Never introduce new actions other than the ones provided.
\n{format_instructions}\n
Objective: {objective}"""

REWRITER ="""\
My objective was: {objective}\n
Original plan to achieve this objective was: {initial_plan}\n
But there are errors in the original plan: {errors}\n
Create a new step-by-step plan to achieve my objective. Remember that you can only use the followings tools in the plan:
{tools} 

Each step in the plan is one dictionary with 3 keys: tool, input_schema and output_key. \
A step in the plan can receive the output from a previous step as input. Use <> to enclose an input when using an output from a previous step.\
Separate two function calls to two steps. Do NOT combine them into one step. \
You must conform to the input specs defined in the tool descriptions. 

You MUST use the following format for the new plan to achieve my objective. You MUST NOT include any other text. \
The output format is {{"steps": [{{[{{tool: str, input_schema: {{input_schema}}, output_key: str}}]}}]}}

Objective can be time sensitive. Pay attention to today's date when composing the plan. \
Today's date is: {date} \n
OBJECTIVE: {objective}\n
"""

SEQUENTIAL_PLANNER_PROMPT = """\
Given the OBJECTIVE, create a step-by-step plan by only using the tools listed below. \
The steps in the plan is a list of dictionaries. Each step is one dictionary with 3 keys: tool, input_schema and output_key. \
A step in the plan can receive the output from a previous step as input. Use <> to enclose an input when using an output from a previous step.\
Separate two function calls to two steps. Do NOT combine them into one step. \
You must conform to the input specs defined in the tool descriptions. 

Output format example:
{{"steps":
    [
        {{
          "tool": "ticker_lookup",
          "input_schema": {{
            "entity": "Apple Inc."
          }},
          "ouput_key": "result_1"
        }},
        {{
          "tool": "get_info",
          "input_schema": {{
            "ticker": "<result_1>"
          }},
          "output_key": "result_2"
        }}
    ]
}}

Tools: 
{tools}

You MUST strictly follow the output format below. You MUST NOT include any other text.
{{  
  "steps": [
    {{"tool": str, "input_schema": {{input_schema}}, "output_key": str}}
  ]
}}
OBJECTIVE: {objective}\n
Objective can be time sensitive. Pay attention to today's date when composing the plan. \
Today's date is: {date} \n
YOUR OUTPUT:
"""