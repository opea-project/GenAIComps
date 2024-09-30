# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the following workers: {members}."
    " The goal is to execute the workflow, then retrieve the results and answer the user's question."
    " First schedule the workflow, once it has been started, continuously check its status until the workflow is finished. Use another tool to retrieve the data. Once you have retrieved the data, then only answer the user's original question."
    " Each worker will perform a task and respond with their results and status."
    " When finished, respond with FINISH."
)

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt + " Given the conversation and history below, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

agent_sys_prompt = (
    "You are a helpful AI assistant, collaborating with other assistants."
    " If you are unable to answer, another assistant with different tools will help."
    " Remember to only give a summary."
    " Do not give your own reasoning."
)
