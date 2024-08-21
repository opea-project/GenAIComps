import unittest
import requests
import base64
import os

class TestLVMService(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://localhost:9399/v1/lvm"

    def test_lvm_generation_with_image_url(self):
        payload = {
            "image": "https://pbs.twimg.com/media/GKLN4qPXEAArqoK.png",  # Replace with a valid URL
            "prompt": "Describe this image.",
            "max_new_tokens": 50,
            "top_k": 50,
            "top_p": 0.99,
            "temperature": 1.0
        }
        response = requests.post(self.base_url, json=payload)
        print("Response:", response.status_code, response.text)  # Debugging print statement
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("text", response_json)
        self.assertTrue(len(response_json["text"]) > 0)

    def test_lvm_generation_with_base64_image(self):
        image_path = "sample.jpg"  # Replace with the correct image path
        if not os.path.exists(image_path):
            self.skipTest(f"Skipping test as {image_path} does not exist.")
        
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        payload = {
            "image": encoded_string,
            "prompt": "Describe this base64 encoded image.",
            "max_new_tokens": 50,
            "top_k": 50,
            "top_p": 0.99,
            "temperature": 1.0
        }
        response = requests.post(self.base_url, json=payload)
        print("Response:", response.status_code, response.text)  # Debugging print statement
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("text", response_json)
        self.assertTrue(len(response_json["text"]) > 0)

    def test_invalid_image_url(self):
        payload = {
            "image": "invalid_url",
            "prompt": "Describe this image.",
            "max_new_tokens": 50,
            "top_k": 50,
            "top_p": 0.99,
            "temperature": 1.0
        }
        response = requests.post(self.base_url, json=payload)
        print("Response:", response.status_code, response.text)  # Debugging print statement
        self.assertEqual(response.status_code, 500)  # Expecting a 500 error due to invalid image URL
        if response.status_code == 500:
            print("500 Error Received - skipping JSON parsing.")
        else:
            try:
                response_json = response.json()
                print("Full Response JSON:", response_json)  # Detailed response print
            except requests.exceptions.JSONDecodeError:
                self.fail("Response is not valid JSON")

    def test_empty_prompt(self):
        payload = {
            "image": "https://pbs.twimg.com/media/GKLN4qPXEAArqoK.png",  # Replace with a valid URL
            "prompt": "",
            "max_new_tokens": 50,
            "top_k": 50,
            "top_p": 0.99,
            "temperature": 1.0
        }
        response = requests.post(self.base_url, json=payload)
        print("Response:", response.status_code, response.text)  # Debugging print statement
        self.assertEqual(response.status_code, 500)  # Expecting a 500 error due to empty prompt
        if response.status_code == 500:
            print("500 Error Received - skipping JSON parsing.")
        else:
            try:
                response_json = response.json()
                print("Full Response JSON:", response_json)  # Detailed response print
            except requests.exceptions.JSONDecodeError:
                self.fail("Response is not valid JSON")

if __name__ == "__main__":
    unittest.main()