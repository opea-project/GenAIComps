# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
import time
import unittest

import requests
from fastapi.responses import StreamingResponse
from prometheus_client import start_http_server

from comps import ServiceOrchestrator, ServiceType, TextDoc, opea_microservices, register_microservice

_METRIC_PORT = 8000


@register_microservice(name="s1", host="0.0.0.0", port=8083, endpoint="/v1/add")
async def s1_add(request: TextDoc) -> TextDoc:
    req = request.model_dump_json()
    req_dict = json.loads(req)
    text = req_dict["text"]
    text += " ~~~"
    return {"text": text}


@register_microservice(name="s0", host="0.0.0.0", port=8085, endpoint="/v1/add", service_type=ServiceType.LLM)
async def s0_add(request: TextDoc) -> TextDoc:
    req = request.model_dump_json()
    req_dict = json.loads(req)
    text = req_dict["text"]

    async def token_generator():
        for i in [" OPEA", " is", " great.", " I", " think ", " so."]:
            yield i

    text += "project!"
    return StreamingResponse(token_generator(), media_type="text/event-stream")


class TestServiceOrchestratorStreaming(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.s0 = opea_microservices["s0"]
        cls.s1 = opea_microservices["s1"]
        cls.process1 = multiprocessing.Process(target=cls.s0.start, daemon=False, name="s0")
        cls.process2 = multiprocessing.Process(target=cls.s1.start, daemon=False, name="s1")
        cls.process1.start()
        cls.process2.start()

        cls.service_builder = ServiceOrchestrator()

        cls.service_builder.add(cls.s0).add(cls.s1)
        cls.service_builder.flow_to(cls.s0, cls.s1)

        # requires prometheus_client >= 0.20.0 (earlier versions return None)
        cls.server, cls.thread = start_http_server(_METRIC_PORT)

    @classmethod
    def tearDownClass(cls):
        cls.s0.stop()
        cls.s1.stop()
        cls.process1.terminate()
        cls.process2.terminate()

        cls.server.shutdown()
        cls.thread.join()

    async def test_schedule(self):
        result_dict, _ = await self.service_builder.schedule(initial_inputs={"text": "hello, "})
        response = result_dict["s1/MicroService"]
        idx = 0
        res_expected = ["OPEA", "is", "great.", "~~~", "I", "think", "so.", "~~~"]
        async for k in response.__reduce__()[2]["body_iterator"]:
            self.assertEqual(self.service_builder.extract_chunk_str(k).strip(), res_expected[idx])
            idx += 1

        r = requests.get(f"http://localhost:{_METRIC_PORT}/metrics", timeout=5)
        self.assertEqual(r.status_code, 200)
        lines = r.text.splitlines()

        metrics = {}
        for line in lines:
            if line.startswith("mega") and "_count" in line:
                items = line.split()
                self.assertTrue(len(items), 2)
                name, value = items

                self.assertTrue(name.startswith("megaservice_"))
                self.assertTrue(name.endswith("_count"))
                metrics[name] = int(float(value))
        print(metrics)

        all_tokens = len(res_expected)
        # After proper orchestrator request processing:
        # - first tokens count should be equal to request count
        # - inter tokens count should not include first token
        correct = {
            "megaservice_request_latency_count": 1,
            "megaservice_first_token_latency_count": 1,
            "megaservice_inter_token_latency_count": all_tokens - 1,
        }
        for name, value in correct.items():
            self.assertTrue(name in metrics)
            self.assertEqual(metrics[name], value)

    def test_extract_chunk_str(self):
        res = self.service_builder.extract_chunk_str("data: [DONE]\n\n")
        self.assertEqual(res, "")
        res = self.service_builder.extract_chunk_str("data: b'example test.'\n\n")
        self.assertEqual(res, "example test.")

    def test_token_generator(self):
        start = time.time()
        sentence = "I write an example test.</s>"
        for i in self.service_builder.token_generator(
            sentence=sentence, token_start=start, is_first=True, is_last=False
        ):
            self.assertTrue(i.startswith("data: b'"))

        for i in self.service_builder.token_generator(
            sentence=sentence, token_start=start, is_first=False, is_last=True
        ):
            self.assertTrue(i.startswith("data: "))


if __name__ == "__main__":
    unittest.main()
