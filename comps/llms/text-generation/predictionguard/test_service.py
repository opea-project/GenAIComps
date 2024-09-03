# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import requests


class TestLLMService(unittest.TestCase):
    def setUp(self):
        # Assuming your service is running locally on port 9000
        self.base_url = "http://localhost:9000/v1/chat/completions"

    def test_llm_generate(self):
        payload = {
            "model": "Hermes-2-Pro-Llama-3-8B",
            "query": "What is the capital of France?",
            "max_new_tokens": 50,
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 50,
            "streaming": False,
        }
        response = requests.post(self.base_url, json=payload)
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("text", response_json)
        self.assertIn("Paris", response_json["text"])

    def test_llm_generate_streaming(self):
        payload = {
            "model": "Hermes-2-Pro-Llama-3-8B",
            "query": "What is the capital of France?",
            "max_new_tokens": 50,
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 50,
            "streaming": True,
        }
        response = requests.post(self.base_url, json=payload, stream=True)
        self.assertEqual(response.status_code, 200)

        # Collecting streamed data
        full_response = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if "data: " in decoded_line:
                    content = decoded_line.replace("data: ", "").strip()
                    full_response += content

        self.assertIn("Paris", full_response)

    def test_empty_query(self):
        payload = {
            "model": "Hermes-2-Pro-Llama-3-8B",
            "query": "",
            "max_new_tokens": 50,
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 50,
            "streaming": False,
        }
        response = requests.post(self.base_url, json=payload)
        self.assertEqual(response.status_code, 500)  # Expecting a 500 error since the query is empty


if __name__ == "__main__":
    unittest.main()
