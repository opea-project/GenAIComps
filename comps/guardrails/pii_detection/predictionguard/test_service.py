import unittest
import requests

class TestPIIGuardService(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://localhost:9080/v1/pii"

    def test_pii_detection(self):
        payload = {
            "prompt": "My name is John Doe, and my credit card number is 4111 1111 1111 1111.",
            "replace": False
        }
        response = requests.post(self.base_url, json=payload)
        print("Response:", response.status_code, response.text)
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("detected_pii", response_json)
        self.assertIsNotNone(response_json["detected_pii"])
        self.assertTrue(len(response_json["detected_pii"]) > 0)

    def test_pii_replacement(self):
        payload = {
            "prompt": "My name is John Doe, and my credit card number is 4111 1111 1111 1111.",
            "replace": True,
            "replace_method": "random"
        }
        response = requests.post(self.base_url, json=payload)
        print("Response:", response.status_code, response.text)
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("new_prompt", response_json)
        self.assertNotEqual(response_json["new_prompt"], payload["prompt"])

    def test_no_pii_detection(self):
        payload = {
            "prompt": "This is a test with no sensitive information.",
            "replace": False
        }
        response = requests.post(self.base_url, json=payload)
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIsNotNone(response_json["detected_pii"])
        self.assertEqual(len(response_json.get("detected_pii", [])), 0)

    def test_invalid_replace_method(self):
        payload = {
            "prompt": "My name is John Doe, and my credit card number is 4111 1111 1111 1111.",
            "replace": True,
            "replace_method": "invalid_method"
        }
        response = requests.post(self.base_url, json=payload)
        print("Response:", response.status_code, response.text)
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()