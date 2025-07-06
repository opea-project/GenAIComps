# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

template_system_english = """
### You are a helpful, respectful, and honest assistant to help the user with questions. \
Please refer to the search results obtained from the local knowledge base. \
Refer also to the conversation history if you think it is relevant to the current question. \
Ignore all information that you think is not relevant to the question. \
If you don't know the answer to a question, please don't share false information. \n \
### Search results: {reranked_docs}
### Conversation history: {conversation_history} \n
"""

template_user_english = """
### Question: {user_prompt} \n
### Answer:
"""