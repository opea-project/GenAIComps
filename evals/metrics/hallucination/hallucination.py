#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
from typing import Dict, Optional, Union

from langchain_community.llms import HuggingFaceEndpoint

from .schema import *
from .template import HallucinationTemplate


class HallucinationMetric:
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str]] = None,
        include_reason: bool = True,
        async_mode: bool = False,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 0 if strict_mode else threshold
        self.model = model
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

    def measure(self, test_case: Dict):
        self.verdicts: List[HallucinationVerdict] = self._generate_verdicts(test_case.actual_output, test_case.context)

        self.score = self._calculate_score()
        self.reason = self._generate_reason()
        self.success = self.score <= self.threshold
        self.verbose_logs = construct_verbose_logs(
            self,
            steps=[
                f"Verdicts:\n{prettify_list(self.verdicts)}",
                f"Score: {self.score}\nReason: {self.reason}",
            ],
        )

        return self.score

    def _generate_reason(self):
        if self.include_reason is False:
            return None

        factual_alignments = []
        contradictions = []
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                factual_alignments.append(verdict.reason)
            else:
                contradictions.append(verdict.reason)

        prompt: dict = HallucinationTemplate.generate_reason(
            factual_alignments=factual_alignments,
            contradictions=contradictions,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = self.model.generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_verdicts(self, actual_output: str, contexts: List[str]) -> List[HallucinationVerdict]:
        verdicts: List[HallucinationVerdict] = []
        prompt = HallucinationTemplate.generate_verdicts(actual_output=actual_output, contexts=contexts)
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            verdicts = [HallucinationVerdict(**item) for item in data["verdicts"]]
            return verdicts
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [HallucinationVerdict(**item) for item in data["verdicts"]]
                return verdicts

    def _calculate_score(self) -> float:
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        hallucination_count = 0
        for verdict in self.verdicts:
            if verdict.verdict.strip().lower() == "no":
                hallucination_count += 1

        score = hallucination_count / number_of_verdicts
        return 1 if self.strict_mode and score > self.threshold else score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score <= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Hallucination"
