# pip install python-arango
# https://github.com/arangodb/python-arango

from arango import ArangoClient
from uuid import uuid4

client = ArangoClient(hosts="http://localhost:8529")

db = client.db("_system", username="root", password="test", verify=True)

if db.has_collection("Prompt"):
    db.delete_collection("Prompt")

db.create_collection("Prompt")

result = db.collection("Prompt").insert({
    "prompt_text": "test prompt",
    "user_text": "test user text"
}, return_new=True)

print(result)