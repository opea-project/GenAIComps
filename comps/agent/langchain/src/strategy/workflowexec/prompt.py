# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

SCHEDULER_SYS_PROMPT = """\
You start the workflow using the parameters provided.

IMPORTANT: You MUST ALWAYS make tool calls.

User request: {input}
Now begin!
"""

STATUS_CHECKER_SYS_PROMPT = """\
You are to check the status of the workflow. The status will be checked repeatedly until the workflow status is finished, so the history might contain multiple tool calls already.
Below shows the previous chat history.

Previous chat history:
{history}
End of the previous chat history.

IMPORTANT: You MUST ALWAYS make tool calls.

Now begin!
"""

DATA_RETRIEVER_SYS_PROMPT = """\
You are to retrieve the data from the workflow.

Previous chat history:
{history}
End of the previous chat history.

IMPORTANT: You MUST ALWAYS make tool calls.

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
