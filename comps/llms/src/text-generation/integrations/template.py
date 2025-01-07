# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re


class ChatTemplate:
    @staticmethod
    def generate_rag_prompt(question, documents, model):
        context_str = "\n".join(documents)
        if model == "meta-llama/Meta-Llama-3.1-70B-Instruct" or model == "meta-llama/Meta-Llama-3.1-8B-Instruct":
            template = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
            Question: {question}
            Context: {context}
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        else:
            if context_str and len(re.findall("[\u4E00-\u9FFF]", context_str)) / len(context_str) >= 0.3:
                # chinese context
                template = """
    ### 你将扮演一个乐于助人、尊重他人并诚实的助手，你的目标是帮助用户解答问题。有效地利用来自本地知识库的搜索结果。确保你的回答中只包含相关信息。如果你不确定问题的答案，请避免分享不准确的信息。
    ### 搜索结果：{context}
    ### 问题：{question}
    ### 回答：
    """
            else:
                template = """
    ### You are a helpful, respectful and honest assistant to help the user with questions. \
    Please refer to the search results obtained from the local knowledge base. \
    But be careful to not incorporate the information that you think is not relevant to the question. \
    If you don't know the answer to a question, please don't share false information. \n
    ### Search results: {context} \n
    ### Question: {question} \n
    ### Answer:
    """
        return template.format(context=context_str, question=question)
