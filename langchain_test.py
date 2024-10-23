from arango import ArangoClient
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.graphs.arangodb_graph import ArangoGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_community.vectorstores.arangodb_vector import ArangoVector
from langchain_core.documents import Document
from langchain_openai import OpenAI

system_db = ArangoClient().db("_system", password="test", verify=True)
system_db.delete_database("langchain_test", ignore_missing=True)
system_db.create_database("langchain_test")
db = ArangoClient().db("langchain_test", password="test", verify=True)

####################
# Test ArangoGraph #
####################

# Create nodes
node1 = Node(id="1", type="Person", properties={"name": "John", "age": 30})
node2 = Node(id="2", type="Person", properties={"name": "Jane", "age": 28})
node3 = Node(id="3", type="Club", properties={"name": "Karate Club"})

# Create relationships
relationship1 = Relationship(source=node1, target=node3, type="MEMBER_OF", properties={"joined_date": "2020-01-01"})
relationship2 = Relationship(source=node2, target=node3, type="MEMBER_OF", properties={"joined_date": "2019-05-15"})
relationship3 = Relationship(source=node1, target=node2, type="KNOWS", properties={"since": "2018-03-10"})

# Create source document
source_doc = Document(
    page_content="John and Jane are members of the Karate Club. They know each other.",
    metadata={"source": "club_records"},
)

# Create GraphDocument
graph_doc = GraphDocument(
    nodes=[node1, node2, node3], relationships=[relationship1, relationship2, relationship3], source=source_doc
)

arango_graph = ArangoGraph(db=db, include_examples=False)
arango_graph.add_graph_documents([graph_doc], graph_name="NewGraph", include_source=True)

#####################
# Test ArangoVector #
#####################

# Add some sample texts
texts = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be, that is the question",
    "All that glitters is not gold",
    "hello what's up",
]

vector_store = ArangoVector.from_texts(
    texts,
    OpenAIEmbeddings(),
    database=db,
    collection_name="vector_test",
    index_name="vector_index",
    distance_strategy="COSINE",
)

texts_2 = ["the dog, cat, and mouse are all mammals"]
vector_store.add_texts(texts_2)

# Perform a similarity search
query = "What animal is mentioned?"
results = vector_store.similarity_search_with_score(query, k=2)

print("Search results for query:", query)
for doc, score in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Score: {score}")
    print("---")

# Try another query
query2 = "What's a famous Shakespeare quote?"
results2 = vector_store.similarity_search_with_score(query2, k=1)

print("\nSearch results for query:", query2)
for doc, score in results2:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Score: {score}")
    print("---")

###########################
# Test ArangoGraphQAChain #
###########################

llm = OpenAI(temperature=0)
graph = ArangoGraph(db=db, include_examples=False, graph_name="NewGraph")
chain = ArangoGraphQAChain.from_llm(llm, graph=graph, allow_dangerous_requests=True)
chain.verbose = True
chain.execute_aql_query = False
chain.run("What is the name of the club?")
chain.execute_aql_query = True
chain.run("What is the name of the club?")
