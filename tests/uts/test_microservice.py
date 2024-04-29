import json
import unittest
from fastapi.testclient import TestClient

from comps import register_microservice, TextDoc, opea_microservices


@register_microservice(name="s1", port=8080, expose_endpoint="/v1/add")
async def add(request: TextDoc) -> TextDoc:
    req = request.json()
    req_dict = json.loads(req)
    text = req_dict["text"]
    text += "OPEA Project!"
    return {"text": text}


class TestMicroService(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(opea_microservices["s1"].app)

        opea_microservices["s1"].start()

    def tearDown(self):
        opea_microservices["s1"].stop()

    def test_add_route(self):
        response = self.client.post("/v1/add", json={"text": "Hello, "})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['text'], "Hello, OPEA Project!")

if __name__ == "__main__":
    unittest.main()