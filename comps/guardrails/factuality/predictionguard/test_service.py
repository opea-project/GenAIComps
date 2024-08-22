# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import requests


class TestFactualityGuardService(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://localhost:9075/v1/factuality"

    def test_factuality_guard_valid(self):
        payload = {
            "reference": "The Earth revolves around the Sun.",
            "text": "The Earth orbits the Sun once every year.",
        }
        response = requests.post(self.base_url, json=payload)
        print("Response:", response.status_code, response.text)  # Debugging print statement
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("score", response_json)
        self.assertTrue(0 <= response_json["score"] <= 1)

    def test_factuality_guard_empty_reference(self):
        payload = {"reference": "", "text": "The Earth orbits the Sun once every year."}
        response = requests.post(self.base_url, json=payload)
        print("Response:", response.status_code, response.text)  # Debugging print statement
        self.assertEqual(response.status_code, 400)  # Expecting a 400 error due to empty reference

    def test_factuality_guard_empty_text(self):
        payload = {"reference": "The Earth revolves around the Sun.", "text": ""}
        response = requests.post(self.base_url, json=payload)
        print("Response:", response.status_code, response.text)  # Debugging print statement
        self.assertEqual(response.status_code, 400)  # Expecting a 400 error due to empty text

    def test_factuality_guard_empty_inputs(self):
        payload = {"reference": "", "text": ""}
        response = requests.post(self.base_url, json=payload)
        print("Response:", response.status_code, response.text)  # Debugging print statement
        self.assertEqual(response.status_code, 400)  # Expecting a 400 error due to both inputs being empty


if __name__ == "__main__":
    unittest.main()
