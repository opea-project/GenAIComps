# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from urllib.parse import quote

import requests

################################################################
#           Download the text file to extract fraph from
################################################################
# Define the input data : big text file and feed it

TEMP_DIR = os.path.join(os.getcwd(), "tmpdata")
FILE_URL = "https://gist.githubusercontent.com/wey-gu/75d49362d011a0f0354d39e396404ba2/raw/0844351171751ebb1ce54ea62232bf5e59445bb7/paul_graham_essay.txt"
command = ["wget", "-P", TEMP_DIR, FILE_URL]
try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print(f"Download successful. Output:\n{result.stdout}")
except subprocess.CalledProcessError as e:
    print(f"Download failed. Error:\n{e.stderr}")


text = open(f"{TEMP_DIR}/paul_graham_essay.txt").read()
encoded_data2 = quote(text)


##################################################################
#   Function to parse the output to decipher if
#   triplets head->relation->tail was extracted
##################################################################
def run_check_keywords(response):
    # Check for keywords in the response
    if all(key in response.text.lower() for key in ["head", "tail", "type"]):
        print("TEST PASS :: All three keys (head, tail, type) exist in the response.")
        return True

    print("TEST FAIL: No keyword found")
    return False


##################################################################
#   Extract graph from text2graph
##################################################################
PORT = 8090
BASE_URL = f"http://localhost:{PORT}/v1/text2graph"
headers = {"accept": "application/json"}

# Send the text as a query parameter
response = requests.post(url=BASE_URL, params={"input_text": text}, headers=headers)
print(f"{response.json()}")
if response.status_code == 200:
    print(f"Microservice response code: {response.status_code}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)

# Check to make sure all works
success = run_check_keywords(response)
# Exit with appropriate status code
sys.exit(0 if success else 1)
