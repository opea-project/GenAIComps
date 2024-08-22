import unittest
import requests

class TestToxicityGuardService(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://localhost:9090/v1/toxicity"

    def test_toxicity_detection_high_score(self):
        payload = {
            "text": "You are an idiot and a loser."
        }
        response = requests.post(self.base_url, json=payload)
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertGreaterEqual(response_json["score"], 0.7)  # Assuming a high score indicates toxicity

    def test_toxicity_detection_low_score(self):
        payload = {
            "text": "Have a great day!"
        }
        response = requests.post(self.base_url, json=payload)
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertLessEqual(response_json["score"], 0.3)  # Assuming a low score indicates non-toxicity

    def test_empty_text_input(self):
        payload = {
            "text": ""
        }
        response = requests.post(self.base_url, json=payload)
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)

    def test_toxicity_with_special_characters(self):
        payload = {
            "text": "@#$%^&*!"
        }
        response = requests.post(self.base_url, json=payload)
        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)
        self.assertGreaterEqual(response.json().get("score", 0), 0.5)  # Assuming special characters might have moderate toxicity

if __name__ == "__main__":
    unittest.main()