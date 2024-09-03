# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


class ChatTemplate:

    @staticmethod
    def generate_multimodal_rag_on_videos_prompt(initial_query: str, retrieved_metadatas):
        context = retrieved_metadatas[0]["transcript_for_inference"]
        template = """The transcript associated with the image is '{context}'. {question}"""
        return template.format(context=context, question=initial_query)
