# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List

from comps import CustomLogger
from comps.cores.proto.docarray import PrevQuestionDetails

logger = CustomLogger(f"{__file__.split('comps/')[1].split('/', 1)[0]}_microservice")


class ConversationHistoryHandler:
    def validate_conversation_history(self, con_history: List[PrevQuestionDetails]):
        if con_history is None:
            return False

        if len(con_history) == 0:
            return False

        if_not_empty_history = False
        for h in con_history:
            if h.question.strip() != "" or h.answer.strip() != "":
                if_not_empty_history = True
        return if_not_empty_history

    def parse_conversation_history(self, con_history: List[PrevQuestionDetails], type: str, params: dict = {}) -> str:
        if self.validate_conversation_history(con_history) is False:
            return ""

        if type.lower() == "naive":
            return self._get_history_naive(con_history, **params)
        else:
            raise ValueError(f"Incorrect ConversationHistoryHandler parsing type. Got: {type}. Expected: [naive, ]")

    def _get_history_naive(self, con_history: List[PrevQuestionDetails], top_k: int = 3) -> str:
        if len(con_history) < top_k:
            last_k_answers = con_history
        else:
            last_k_answers = con_history[-top_k:]

        formatted_output = ""
        for conv in last_k_answers:
            formatted_output += f"User: {conv.question}\nAssistant: {conv.answer}\n"

        logger.info(formatted_output)
        return formatted_output.strip()
