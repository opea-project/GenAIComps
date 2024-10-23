Instructions

0. Create a virtual environment:

```bash
python -m venv .venv

source .venv/bin/activate
```

1. Install the required packages:

```bash
pip install python-arango
pip install langchain_openai
pip install git+https://github.com/arangoml/langchain.git@arangodb#subdirectory=libs/community
```

2. Provision the ArangoDB with Vector Index image:

```bash
docker create --name arango-vector -p 8529:8529 -e ARANGO_ROOT_PASSWORD=test jbajic/arangodb-arm:vector-index-preview

docker start arango-vector
```

3. Set your `OPENAI_API_KEY` environment variable (contact Anthony for access)

4. Run the test script to confirm LangChain is working:

```bash
python langchain_test.py
```