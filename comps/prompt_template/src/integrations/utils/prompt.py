# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def generate_prompt_templates(context: str, question: str) -> tuple[str, str]:
    """Dynamically generates the system and user prompts based on the given context and question.

    Args:
        context (str): The context information to assist in answering the question.
        question (str): The user's question.

    Returns:
        tuple[str, str]: A tuple containing the system prompt and user prompt.
    """
    system_prompt = f"""You are a helpful and knowledgeable assistant. Use the following context to answer the question accurately.

Context:
{context}

"""
    user_prompt = f"""Question:
{question}

Answer:"""

    return system_prompt, user_prompt
