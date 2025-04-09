# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import multiprocessing
import time
import unittest

import requests

from comps import (
    ServiceOrchestrator,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.cores.mega.base_statistics import collect_all_statistics

SVC1 = "opea_service@s1_add"
SVC2 = "open_service@test"


@register_statistics(names=[SVC1])
@register_statistics(names=[SVC2])
@register_microservice(name="s1", host="0.0.0.0", port=8083, endpoint="/v1/add")
async def s1_add(request: TextDoc) -> TextDoc:
    start = time.monotonic()
    time.sleep(5)
    req = request.model_dump_json()
    req_dict = json.loads(req)
    text = req_dict["text"]
    text += "opea"
    statistics_dict[SVC1].append_latency(time.monotonic() - start, None)
    return {"text": text}


class TestBaseStatistics(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.s1 = opea_microservices["s1"]
        self.process1 = multiprocessing.Process(target=self.s1.start, daemon=False, name="s1")
        self.process1.start()

        self.service_builder = ServiceOrchestrator()
        self.service_builder.add(opea_microservices["s1"])

    def tearDown(self):
        self.s1.stop()
        self.process1.terminate()

    async def test_base_statistics(self):
        for _ in range(2):
            task1 = asyncio.create_task(self.service_builder.schedule(initial_inputs={"text": "hello, "}))
            await asyncio.gather(task1)
            _result_dict1, _ = task1.result()

        response = requests.get("http://localhost:8083/v1/statistics")
        res = response.json()
        p50 = res[SVC1]["p50_latency"]
        p99 = res[SVC1]["p99_latency"]
        self.assertEqual(int(p50), int(p99))


class TestBaseStatisticsLocal(unittest.TestCase):
    def test_stats(self):
        stats = statistics_dict[SVC2]

        stats.append_latency(3)
        res = collect_all_statistics()
        avg = res[SVC2]["average_latency"]
        self.assertIsNotNone(avg)
        p99 = res[SVC2]["p99_latency"]
        self.assertEqual(int(p99), int(avg))
        p50 = res[SVC2]["p50_latency"]
        self.assertEqual(int(p99), int(p50))
        self.assertIsNone(res[SVC2]["p50_latency_first_token"])

        stats.append_latency(2, 1)
        res = collect_all_statistics()
        avg = res[SVC2]["average_latency_first_token"]
        self.assertIsNotNone(avg)
        p99 = res[SVC2]["p99_latency_first_token"]
        self.assertEqual(int(p99), int(avg))
        p50 = res[SVC2]["p50_latency_first_token"]
        self.assertEqual(int(p99), int(p50))

        stats.append_latency(1)
        res = collect_all_statistics()
        p50 = res[SVC2]["p50_latency"]
        avg = res[SVC2]["average_latency"]
        self.assertEqual(int(avg), int(p50))
        self.assertEqual(int(p50), 2)


if __name__ == "__main__":
    unittest.main()
