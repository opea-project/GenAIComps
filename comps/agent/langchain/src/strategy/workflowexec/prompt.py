# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

SCHEDULER_SYS_PROMPT = """\
You start the workflow using the parameters provided.

IMPORTANT: You MUST ALWAYS make tool calls. Only call 1 tool per step.

User request: {input}
Now begin!
"""

STATUS_CHECKER_SYS_PROMPT = """\
You are to check the status of the workflow.

Begin Execution History:
{history}
End Execution History.

IMPORTANT: You MUST ALWAYS make tool calls. Only call 1 tool per step.

User request: {input}
Now begin!
"""

DATA_RETRIEVER_SYS_PROMPT = """\
You are to retrieve the data from the workflow.

Begin Execution History:
{history}
End Execution History.

IMPORTANT: You MUST ALWAYS make tool calls. Only call 1 tool per step.

User request: {input}
Now begin!
"""

REASONING_PROMPT = """\
You are a helpful assistant. Use the data provided by the tool output to answer the user's original question.
Give your answer in a one short sentence.

Begin tool output:
{tool_output}
End tool output.

User request: {input}
Now begin!
"""
