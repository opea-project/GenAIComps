# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import requests


class TestEmbeddingService(unittest.TestCase):
    def setUp(self):
        # Assuming your service is running locally on port 6000
        self.base_url = "http://localhost:6000/v1/embeddings"

    def test_embedding_generation(self):
        response = requests.post(self.base_url, json={"text": "Hello, world!"})
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("embedding", response_json)
        self.assertIsInstance(response_json["embedding"], list)

    def test_empty_input(self):
        response = requests.post(self.base_url, json={"text": ""})
        self.assertEqual(response.status_code, 400)
        response_json = response.json()
        self.assertIn("detail", response_json)
        self.assertEqual(response_json["detail"], "Input text cannot be empty")

    def test_embedding_vector_length(self):
        response = requests.post(
            self.base_url, json={"text": "This is a test text to verify the length of the embedding vector."}
        )
        self.assertEqual(response.status_code, 200)
        embedding = response.json().get("embedding")
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 512)  # Ensuring the embedding is truncated to 512 elements


if __name__ == "__main__":
    unittest.main()
