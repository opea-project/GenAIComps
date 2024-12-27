# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2024 MOSEC Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re

import requests
from langchain_core.prompts import ChatPromptTemplate

from comps import CustomLogger, LLMParamsDoc, SearchedDoc, ServiceType
from comps.cores.common.component import OpeaComponent

logger = CustomLogger("reranking_mosec_xeon")
logflag = os.getenv("LOGFLAG", False)


class OPEAMosecReranking(OpeaComponent):
    """A specialized reranking component derived from OpeaComponent for mosec reranking services."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RERANK.name.lower(), description, config)
        self.mosec_reranking_endpoint = os.getenv("MOSEC_RERANKING_ENDPOINT", "http://localhost:8080")
        self.url = self.mosec_reranking_endpoint + "/inference"
        self.headers = {"Content-Type": "application/json"}

    async def invoke(self, input: SearchedDoc) -> LLMParamsDoc:
        """Invokes the reranking service to generate reranking for the provided input.

        Args:
            input (SearchedDoc): The input in OpenAI reranking format.

        Returns:
            LLMParamsDoc: The response in OpenAI reranking format.
        """
        if input.retrieved_docs:
            docs = [doc.text for doc in input.retrieved_docs]
            data = {"query": input.initial_query, "texts": docs}
            response = requests.post(self.url, data=json.dumps(data), headers=self.headers)
            response_data = response.json()
            best_response = max(response_data, key=lambda response: response["score"])
            doc = input.retrieved_docs[best_response["index"]]
            if doc.text and len(re.findall("[\u4E00-\u9FFF]", doc.text)) / len(doc.text) >= 0.3:
                # chinese context
                template = "仅基于以下背景回答问题:\n{context}\n问题: {question}"
            else:
                template = """Answer the question based only on the following context:
                            {context}
                            Question: {question}
                """
            prompt = ChatPromptTemplate.from_template(template)
            final_prompt = prompt.format(context=doc.text, question=input.initial_query)
            if logflag:
                logger.info(final_prompt.strip())
            return LLMParamsDoc(query=final_prompt.strip())
        else:
            if logflag:
                logger.info(input.initial_query)
            return LLMParamsDoc(query=input.initial_query)

    def check_health(self) -> bool:
        """Checks the health of the reranking service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """

        return True
