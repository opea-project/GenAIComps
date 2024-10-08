# Workflow Executor Agent

## Description

This agent strategy is designed to handle running workflow operation tools. The strategy includes the following steps:

1. `WorkflowScheduler` - Invokes llm to generate `workflow_scheduler` tool call to start workflow with extracted parameters and workflow id from user query.
    - Executes `workflow_scheduler` tool in `ToolChainNode`.
    - `ToolResult` will contain a `workflow_key` used to track current workflow for `WorkflowStatusChecker` and `WorkflowDataRetriever`.
    - After starting workflow, go to `WorkflowStatusChecker`.

2. `WorkflowStatusChecker` - Invokes llm to generate `workflow_status_checker` tool call to check workflow execution status.
    - Executes `workflow_status_checker` tool in `ToolChainNode`.
    - Repeat step until `workflow_status` returned from tool is `finished`.
    - If max number of retries are exceeded before status returns `finished`, go to `END`.
    - When `workflow_status` returns `finished`, go to `WorkflowDataRetriever`.

3. `WorkflowDataRetriever` - Invokes llm to generate `workflow_data_retriever` tool call to retrieve the output data from the workflow.
    - Executes `workflow_data_retriever` tool in `ToolChainNode`.
    - Sends `state` which now contains retrieved data to `ReasoningNode`.

4. `ToolChainNode` - Incorporates `ToolNode` to execute tool calls passed from `WorkflowScheduler`, `WorkflowStatusChecker`, and `WorkflowDataRetriever`.

5. `ReasoningNode` - Used to answer  the user's original question with the provided workflow output data from `WorkflowDataRetriever`.

    - After reasoning, go to `END`.
    - The reasoning agent prompt can be customized to obtain a desired final output response.

Below shows the flow of the workflow executor:

![image](https://github.com/user-attachments/assets/3990669d-d48a-49e4-aecf-069872255a1d)

## Workflow Diagram

Here's what the Langgraph workflow diagram looks like:

![image](https://github.com/user-attachments/assets/ce6ed420-9431-4e5f-9628-b97f4ccdb252)

## Note

- This strategy has only been tested with `mistralai/Mistral-7B-Instruct-v0.3` model.
