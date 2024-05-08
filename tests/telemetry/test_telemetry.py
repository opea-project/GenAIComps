from comps import opea_telemetry
import unittest
import time
import asyncio
import subprocess
from comps.telemetry.opea_telemetry import in_memory_exporter

@opea_telemetry
def dummy_func():
    time.sleep(1)

@opea_telemetry
async def dummy_async_func():
    await asyncio.sleep(2)


class TestTelemetry(unittest.TestCase):
    def setUp(self):
        self.p = subprocess.Popen(
            f"docker run -d -p 4317:4317 -p 4318:4318 --rm -v $(pwd)/collector-config.yaml:/etc/otelcol/config.yaml otel/opentelemetry-collector",
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,)
        raw_output, raw_err = self.p.communicate()
        self.cid = raw_output.decode().replace("\n","")

    def tearDown(self):
        rp = subprocess.Popen(
            f"docker rm -f {self.cid}",
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,)
        _, _ = rp.communicate()
        rp.kill()
        self.p.kill()


    def test_time_tracing(self):
        dummy_func()
        asyncio.run(dummy_async_func())
        time.sleep(2)   # wait until all telemetries are finished
        self.assertEqual(len(in_memory_exporter.get_finished_spans()), 2)
        for i in in_memory_exporter.get_finished_spans():
            if i.name == 'dummy_func':
                self.assertTrue(int((i.end_time - i.start_time) / 1e9) == 1)
            elif i.name == 'dummy_async_func':
                self.assertTrue(int((i.end_time - i.start_time) / 1e9) == 2)
            else:
                self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
