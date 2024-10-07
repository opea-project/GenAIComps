# Workflow Executor Agent

## Description

This agent strategy is designed to handle running workflow operation tools. The strategy includes the following steps:

1. `WorkflowScheduler` - Performs a tool call to start workflow.

- After obtaining the `workflow_key` from `ToolResult`, sends the `state` to `WorkflowStatusChecker`.

2. `WorkflowStatusChecker` - Performs a tool call to check the workflow execution status.

- This step will be repeated until the `workflow_status` returned is `finished`, or the max number of retries are exceeded.
- When `workflow_status` is `finished`, sends `state` to `WorkflowDataRetriever`.

3. `WorkflowDataRetriever` - Performs a tool call to retrieve the output data from the workflow.

- Sends the retrieved data to `reasoning_agent`.

4. `tool_node` - Function that incorporates `ToolNode` to execute tool calls passed from the above 3 classes.

5. `reasoning_agent` - Used to answer the user's original question with the provided workflow output data from `WorkflowDataRetriever`.

- After reasoning, END.
- The reasoning agent prompt can be customized to obtain a desired final output response.

## Workflow Diagram

Here's what the Langgraph workflow diagram looks like:

![image](https://github.com/user-attachments/assets/ce6ed420-9431-4e5f-9628-b97f4ccdb252)

## Note

- The strategy of this workflow has only been tested with `mistralai/Mistral-7B-Instruct-v0.3` model.
