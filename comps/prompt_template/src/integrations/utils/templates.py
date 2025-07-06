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

template_002_chinese = """
### 你将扮演一个乐于助人、尊重他人并诚实的助手，你的目标是帮助用户解答问题。有效地利用来自本地知识库的搜索结果。确保你的回答中只包含相关信息。如果你不确定问题的答案，请避免分享不准确的信息。
### 搜索结果：{reranked_docs}
### 问题：{initial_query}
### 回答：
"""
