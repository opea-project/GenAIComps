# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from typing import Literal
from urllib.parse import quote

import requests

print("----------- Extract graph from CSV -----------")

print("----------- Generating index ----------------")
os.environ["LOAD_FORMAT"] = "CSV"
load_format = os.getenv("LOAD_FORMAT")
CYPHER_CSV_CMD = "LOAD CSV WITH HEADERS FROM 'file:////test1.csv' AS row \
                CREATE (:Person {ID: toInteger(row.ID), Name: row.Name, Age: toInteger(row.Age), City: row.City});"

print(f" CYPHER COMMAND USED:: {CYPHER_CSV_CMD} ")
STRUCT2GRAPH_PORT = os.getenv("STRUCT2GRAPH_PORT")
url = f"http://localhost:{STRUCT2GRAPH_PORT}/v1/struct2graph"
headers = {"accept": "application/json", "Content-Type": "application/json"}

payload = {"input_text": "", "task": "Index", "cypher_cmd": CYPHER_CSV_CMD}

try:
    # Send the POST request
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an exception for bad status codes
    print("Request successful:", response.json())
except requests.exceptions.RequestException as e:
    print("Request failed:", e)

print("----------- Loading graph completed Query ----------------")
print("----------- Issuing Query --------------------------------")
payload = {"input_text": "MATCH (p:Person {Name:'Alice'}) RETURN p", "task": "Query", "cypher_cmd": ""}

try:
    # Send the POST request
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an exception for bad status codes
    print("Request successful:", response.json())
except requests.exceptions.RequestException as e:
    print("Request failed:", e)


print("----------- Extract graph from JSON -----------")

print("----------- Generating index ----------------")

os.environ["LOAD_FORMAT"] = "JSON"
load_format = os.getenv("LOAD_FORMAT")
CYPHER_JSON_CMD = " \
CALL apoc.load.json('file:///test1.json') YIELD value \
UNWIND value.table AS row \
CREATE (:Person { \
          ID: row.ID, \
          Name: row.Name, \
          Age: row.Age, \
          City: row.City \
       }); \
 "

print(f" CYPHER COMMAND USED:: {os.environ['CYPHER_JSON_CMD']}")

url = f"http://localhost:{STRUCT2GRAPH_PORT}/v1/struct2graph"
headers = {"accept": "application/json", "Content-Type": "application/json"}

payload = {"input_text": "", "task": "Index", "cypher_cmd": CYPHER_JSON_CMD}

try:
    # Send the POST request
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an exception for bad status codes
    print("Request successful:", response.json())
except requests.exceptions.RequestException as e:
    print("Request failed:", e)


print("----------- Loading graph completed Query ----------------")
print("----------- Issuing Query ----------------")
payload = {"input_text": "MATCH (n) RETURN n", "task": "Query", "cypher_cmd": ""}

try:
    # Send the POST request
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an exception for bad status codes
    print("Request successful:", response.json())
except requests.exceptions.RequestException as e:
    print("Request failed:", e)


print("----------- Issuing Query ----------------")
payload = {"input_text": "MATCH (p:Person {Name:'Alice'}) RETURN p", "task": "Query", "cypher_cmd": ""}

print(f" Issuing query {payload}")

response = requests.post(url, headers=headers, json=payload)
print(f"RESULT : {response}")
# Get response details
print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.json()}")
