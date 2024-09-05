# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import unittest

from comps import Gateway, ServiceOrchestrator, TextDoc, opea_microservices, register_microservice
from comps import (
        MultimodalDoc, 
        EmbedDoc, 
        EmbedMultimodalDoc, 
        SearchedMultimodalDoc,
        LVMSearchedMultimodalDoc,
        MultimodalRAGQnAWithVideosGateway
    )


@register_microservice(name="mm_embedding", host="0.0.0.0", port=8083, endpoint="/v1/mm_embedding")
async def mm_embedding_add(request: MultimodalDoc) -> EmbedDoc:
    req = request.model_dump_json()
    req_dict = json.loads(req)
    print('req_dict_embedding', req_dict)
    text = req_dict["text"]
    res = {}
    res["text"] = text
    res["embedding"] = [0.12, 0.45]
    return res


@register_microservice(name="mm_retriever", host="0.0.0.0", port=8084, endpoint="/v1/mm_retriever")
async def mm_retriever_add(request: EmbedMultimodalDoc) -> SearchedMultimodalDoc:
    req = request.model_dump_json()
    req_dict = json.loads(req)
    print('req_dict_retriever', req_dict)
    text = req_dict["text"]
    res = {}
    res['retrieved_docs'] = []
    res['initial_query'] = text
    res['top_n'] = 1
    res['metadata'] = [{
            "b64_img_str": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC", 
            "transcript_for_inference": "yellow image"
        }]
    res['chat_template'] = "The caption of the image is: '{context}'. {question}"
    return res

@register_microservice(name="lvm", host="0.0.0.0", port=8085, endpoint="/v1/lvm")
async def lvm_add(request: LVMSearchedMultimodalDoc) -> TextDoc:
    req = request.model_dump_json()
    req_dict = json.loads(req)
    print('req_dict_lvm', req_dict)
    text = req_dict["initial_query"]
    text += "opea project!"
    res = {}
    res['text'] = text
    return res


class TestServiceOrchestrator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mm_embedding = opea_microservices["mm_embedding"]
        self.mm_retriever = opea_microservices["mm_retriever"]
        self.lvm = opea_microservices["lvm"]
        self.mm_embedding.start()
        self.mm_retriever.start()
        self.lvm.start()

        self.service_builder = ServiceOrchestrator()

        self.service_builder.add(opea_microservices["mm_embedding"]).add(opea_microservices["mm_retriever"]).add(opea_microservices["lvm"])
        self.service_builder.flow_to(self.mm_embedding, self.mm_retriever)
        self.service_builder.flow_to(self.mm_retriever, self.lvm)

        self.follow_up_query_service_builder = ServiceOrchestrator()
        self.follow_up_query_service_builder.add(self.lvm)

        self.gateway = MultimodalRAGQnAWithVideosGateway(self.service_builder, self.follow_up_query_service_builder, port=9898)

    def tearDown(self):
        self.mm_embedding.stop()
        self.mm_retriever.stop()
        self.lvm.stop()
        self.gateway.stop()

    async def test_schedule(self):
        result_dict, _ = await self.service_builder.schedule(initial_inputs={"text": "hello, "})
        self.assertEqual(result_dict[self.lvm.name]["text"], "hello, opea project!")


if __name__ == "__main__":
    unittest.main()
