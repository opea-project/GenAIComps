# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import collections
import json
import logging
from enum import Enum
from typing import Dict, List, Optional, Union

import requests
from requests.exceptions import RequestException
from rogue import Rogue

from .template import SummarizationTemplate

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

LLM_JUDGE_METRICS = {
    "Relevance": SummarizationTemplate.generate_relevance,
    "Coherence": SummarizationTemplate.generate_coherence,
    "Consistency": SummarizationTemplate.generate_consistency,
    "Fluency": SummarizationTemplate.generate_fluency,
}


class SummarizationMetric:
    """The summarization metric not only uses your LLMs (application) to generate summaries for evaluation,
    but also uses LLMs to judge whether your LLM (application) is generating Relevance,
    Coherence, Consistency, Fluency summaries."""

    def __init__(
        self,
        model: Optional[Union[str]] = None,
        llm_judge: Optional[Union[str]] = None,
    ):
        """
        Args:
            model: your LLMs endpoint (application) to generate summaries
            llm_judge: LLMs endpoint for judge summaries
        """

        self.model = model
        self.headers = {"Content-Type": "application/json"}
        self.llm_judge = llm_judge
        self.metrics = collections.defaultdict(list)
        self.rogue = Rogue()

    def rouge_scores(self, text1, text2):
        eval_rouge = self.rogue.get_scores(text1, text2)
        self.metrics["rouge-1|F-Score"].append(eval_rouge[0]["rouge-1"]["f"])
        self.metrics["rouge-2|F-Score"].append(eval_rouge[0]["rouge-2"]["f"])
        self.metrics["rouge-l|F-Score"].append(eval_rouge[0]["rouge-l"]["f"])

    def llm_scores(self, document, summary):
        for metric in LLM_JUDGE_METRICS:
            req = {
                "inputs": LLM_JUDGE_METRICS[metric](document, summary),
                "parameters": {"max_new_tokens": 5, "do_sample": False},
            }

            try:
                response = requests.post(
                    f"{self.llm_judge}",
                    headers=self.headers,
                    data=json.dumps(req),
                )
                response.raise_for_status()
                response = response.json()
            except RequestException as e:
                logger.info(str(e))
                continue

            score = response["generated_text"].strip()
            self.metrics[metric].append(int(score))

    def summarize(self, document: str, ref_summary: str, **generation_kwargs):
        req = {"inputs": SummarizationTemplate.generate_summary(document), "parameters": generation_kwargs}

        try:
            response = requests.post(
                f"{self.model}",
                headers=self.headers,
                data=json.dumps(req),
            )
            response.raise_for_status()
            response = response.json()
        except RequestException as e:
            logger.info(str(e))

        gen_summary = response["generated_text"]

        # get metrics
        self.rouge_scores(gen_summary, ref_summary)
        if self.llm_judge is not None:
            self.llm_scores(document, gen_summary)

    @property
    def average_score(self):
        return {metric: sum(self.metrics[metric]) / len(self.metrics[metric]) for metric in self.metrics}
