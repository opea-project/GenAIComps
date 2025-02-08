import os
import time
from typing import Any, Union

import openai
from arango import ArangoClient
from config import (
    SUMMARIZER_ENABLED,
    ARANGO_DB_NAME,
    ARANGO_DISTANCE_STRATEGY,
    ARANGO_EMBEDDING_FIELD,
    ARANGO_GRAPH_NAME,
    ARANGO_NUM_CENTROIDS,
    ARANGO_PASSWORD,
    ARANGO_TEXT_FIELD,
    ARANGO_TRAVERSAL_ENABLED,
    ARANGO_TRAVERSAL_MAX_DEPTH,
    ARANGO_URL,
    ARANGO_USE_APPROX_SEARCH,
    ARANGO_USERNAME,
    HUGGINGFACEHUB_API_TOKEN,
    OPENAI_API_KEY,
    OPENAI_EMBED_MODEL,
    TEI_EMBED_MODEL,
    TEI_EMBEDDING_ENDPOINT,
    ARANGO_TRAVERSAL_MAX_RETURNED,
    OPENAI_CHAT_ENABLED,
    OPENAI_CHAT_TEMPERATURE,
    OPENAI_CHAT_MODEL,
    TGI_LLM_ENDPOINT,
    TGI_LLM_MAX_NEW_TOKENS,
    TGI_LLM_TEMPERATURE,
    TGI_LLM_TIMEOUT,
    TGI_LLM_TOP_K,
    TGI_LLM_TOP_P,
)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.vectorstores.arangodb_vector import ArangoVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint

from comps import (
    CustomLogger,
    EmbedDoc,
    SearchedDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.cores.proto.api_protocol import (
    ChatCompletionRequest,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResponseData,
)

logger = CustomLogger("retriever_arango")
logflag = os.getenv("LOGFLAG", True)

@OpeaComponentRegistry.register("OPEA_RETRIEVER_ARANGO")
class OpeaOpensearchRetriever(OpeaComponent):
    """A specialized retriever component derived from OpeaComponent for ArangoDB retriever services.

    Attributes:
        client (ArangoDB): An instance of the ArangoDB client for vector database operations.
    """
    def __init__(self):
        self.initialize_llm()
        self.initialize_arangodb()

    def initialize_llm(self):
        if OPENAI_API_KEY and OPENAI_CHAT_ENABLED:
            if logflag:
                logger.info("OpenAI API Key is set. Verifying its validity...")
            openai.api_key = OPENAI_API_KEY

            try:
                openai.models.list()
                if logflag:
                    logger.info("OpenAI API Key is valid.")
                self.llm = ChatOpenAI(temperature=OPENAI_CHAT_TEMPERATURE, max_tokens=512, model_name=OPENAI_CHAT_MODEL)
            except openai.error.AuthenticationError:
                if logflag:
                    logger.info("OpenAI API Key is invalid.")
            except Exception as e:
                if logflag:
                    logger.info(f"An error occurred while verifying the API Key: {e}")
        elif TGI_LLM_ENDPOINT:
            self.llm = HuggingFaceEndpoint(
                endpoint_url=TGI_LLM_ENDPOINT,
                max_new_tokens=TGI_LLM_MAX_NEW_TOKENS,
                top_k=TGI_LLM_TOP_K,
                top_p=TGI_LLM_TOP_P,
                temperature=TGI_LLM_TEMPERATURE,
                timeout=TGI_LLM_TIMEOUT,
            )
        else:
            raise ValueError("No text generation environment variables are set, cannot generate graphs.")

    def initialize_arangodb(self):
        self.client = ArangoClient(hosts=ARANGO_URL)
        sys_db = self.client.db(name="_system", username=ARANGO_USERNAME, password=ARANGO_PASSWORD, verify=True)

        if not sys_db.has_database(ARANGO_DB_NAME):
            sys_db.create_database(ARANGO_DB_NAME)

        self.db = self.client.db(name=ARANGO_DB_NAME, username=ARANGO_USERNAME, password=ARANGO_PASSWORD, verify=True)

        if logflag:
            logger.info(f"Connected to ArangoDB {self.db.version()}.")

    def check_health(self) -> bool:
        """Checks the health of the retriever service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if logflag:
            logger.info("[ check health ] start to check health of milvus")
        try:
            # _ = self.client.client.list_collections()
            if logflag:
                logger.info("[ check health ] Successfully connected to ArangoDB!")
            return True
        except Exception as e:
            logger.info(f"[ check health ] Failed to connect to ArangoDB: {e}")
            return False

    def fetch_neighborhoods(self, vector_db, keys, neighborhoods, graph_name, source_collection_name, max_depth):
        if max_depth <= 0:
            start_vertex = "v1"
            links_to_query = ""
        else:
            start_vertex = "v2"
            links_to_query = f"FOR v2 IN 1..{max_depth} ANY v1 {graph_name}_LINKS_TO OPTIONS {{uniqueEdges: 'path'}}"
        aql = f"""
            FOR doc IN @@collection
                FILTER doc._key IN @keys

                LET source_neighborhood = (
                    FOR v1 IN 1..1 INBOUND doc {graph_name}_HAS_SOURCE
                        {links_to_query}
                            FOR s IN 1..1 OUTBOUND {start_vertex} {graph_name}_HAS_SOURCE
                                FILTER s._key != doc._key

                                // Top 3 Most Relevant Chunks (Deterministic)
                                LET score = COSINE_SIMILARITY(doc.{ARANGO_EMBEDDING_FIELD}, s.{ARANGO_EMBEDDING_FIELD})
                                SORT score DESC
                                LIMIT {ARANGO_TRAVERSAL_MAX_RETURNED}

                                COLLECT id = s._key, text = s.{ARANGO_TEXT_FIELD}
                                RETURN {{[id]: text}}
                )

                RETURN {{[doc._key]: source_neighborhood}}
        """

        bind_vars = {
            "@collection": source_collection_name,
            "keys": keys,
        }

        cursor = vector_db.db.aql.execute(aql, bind_vars=bind_vars)

        for doc in cursor:
            neighborhoods.update(doc)

        if logflag:
            logger.info(f"Fetched neighborhoods for {len(neighborhoods)} documents.")

    async def invoke(self, input: Union[EmbedDoc, RetrievalRequest, ChatCompletionRequest]) -> list:
        logger.info(input)

        if isinstance(input, EmbedDoc):
            empty_result = SearchedDoc(retrieved_docs=[], initial_query=input.text)
        elif isinstance(input, RetrievalRequest):
            empty_result = RetrievalResponse(retrieved_docs=[])
        elif isinstance(input, ChatCompletionRequest):
            input.retrieved_docs = []
            input.documents = []
            empty_result = input
        else:
            raise ValueError("Invalid input type: ", type(input))

        start = time.time()

        query = input.text if isinstance(input, EmbedDoc) else input.input
        embedding = input.embedding if isinstance(input.embedding, list) else None

        graph_name = None
        query_split = query.split("|")

        if len(query_split) == 2:
            query = query_split[0].strip()
            graph_name = query_split[1].strip()

        if not graph_name:
            graph_name = ARANGO_GRAPH_NAME

        source_collection_name = f"{graph_name}_SOURCE"

        if not self.db.has_graph(graph_name):
            if logflag:
                graph_names = [g["name"] for g in self.db.graphs()]
                logger.error(f"Graph '{graph_name}' does not exist in ArangoDB. Graphs: {graph_names}")

            return empty_result

        if not self.db.graph(graph_name).has_vertex_collection(source_collection_name):
            if logflag:
                collection_names = self.db.graph(graph_name).vertex_collections()
                m = f"Collection '{source_collection_name}' does not exist in graph '{graph_name}'. Collections: {collection_names}"
                logger.error(m)

            return empty_result

        collection = self.db.collection(source_collection_name)

        collection_count = collection.count()
        if collection_count == 0:
            if logflag:
                logger.error(f"Collection '{source_collection_name}' is empty.")

            return empty_result

        if collection_count < ARANGO_NUM_CENTROIDS:
            if logflag:
                m = f"Collection '{source_collection_name}' has fewer documents ({collection_count}) than the number of centroids ({ARANGO_NUM_CENTROIDS})."
                logger.error(m)

            return empty_result

        random_doc = collection.random()
        random_doc_id = random_doc["_id"]

        embedding = random_doc.get(ARANGO_EMBEDDING_FIELD)

        if not embedding:
            if logflag:
                logger.error(f"Document '{random_doc_id}' is missing field '{ARANGO_EMBEDDING_FIELD}'.")

            return empty_result

        if not isinstance(embedding, list):
            if logflag:
                logger.error(f"Document '{random_doc_id}' has a non-list embedding field, found {type(embedding)}.")

            return empty_result

        dimension = len(embedding)

        if dimension == 0:
            if logflag:
                logger.error(f"Document '{random_doc_id}' has an empty embedding field.")

            return empty_result

        if OPENAI_API_KEY and OPENAI_EMBED_MODEL:
            embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, dimensions=dimension)
        elif TEI_EMBEDDING_ENDPOINT and HUGGINGFACEHUB_API_TOKEN:
            embeddings = HuggingFaceHubEmbeddings(
                model=TEI_EMBEDDING_ENDPOINT, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
            )
        else:
            embeddings = HuggingFaceBgeEmbeddings(model_name=TEI_EMBED_MODEL)

        vector_db = ArangoVector(
            embedding=embeddings,
            embedding_dimension=dimension,
            database=self.db,
            collection_name=source_collection_name,
            embedding_field=ARANGO_EMBEDDING_FIELD,
            text_field=ARANGO_TEXT_FIELD,
            distance_strategy=ARANGO_DISTANCE_STRATEGY,
            num_centroids=ARANGO_NUM_CENTROIDS,
        )

        try:
            if input.search_type == "similarity_score_threshold":
                docs_and_similarities = await vector_db.asimilarity_search_with_relevance_scores(
                    query=query,
                    embedding=embedding,
                    k=input.k,
                    score_threshold=input.score_threshold,
                    use_approx=ARANGO_USE_APPROX_SEARCH,
                )
                search_res = [doc for doc, _ in docs_and_similarities]
            elif input.search_type == "mmr":
                search_res = await vector_db.amax_marginal_relevance_search(
                    query=query,
                    embedding=embedding,
                    k=input.k,
                    fetch_k=input.fetch_k,
                    lambda_mult=input.lambda_mult,
                    use_approx=ARANGO_USE_APPROX_SEARCH,
                )
            else:
                search_res = await vector_db.asimilarity_search(
                    query=query,
                    embedding=embedding,
                    k=input.k,
                    use_approx=ARANGO_USE_APPROX_SEARCH,
                )
        except Exception as e:
            if logflag:
                logger.error(f"Error during similarity search: {e}")

            return empty_result

        if not search_res:
            if logflag:
                logger.info("No documents found.")
            return empty_result

        neighborhoods = {}
        if logflag:
            logger.info(f"ARANGO_TRAVERSAL_ENABLED: {ARANGO_TRAVERSAL_ENABLED}")
            logger.info(f"Number of search results to traverse: {len([r.id for r in search_res])}")

        if ARANGO_TRAVERSAL_ENABLED:
            try:
                if logflag:
                    logger.info(f"Starting neighborhood fetch for keys: {[r.id for r in search_res]}")
                    logger.info(f"Graph name: {graph_name}")
                    logger.info(f"Source collection: {source_collection_name}")
                    logger.info(f"Max depth: {ARANGO_TRAVERSAL_MAX_DEPTH}")

                self.fetch_neighborhoods(
                    vector_db=vector_db,
                    keys=[r.id for r in search_res],
                    neighborhoods=neighborhoods,
                    graph_name=graph_name,
                    source_collection_name=source_collection_name,
                    max_depth=ARANGO_TRAVERSAL_MAX_DEPTH,
                )

                if logflag:
                    logger.info(f"Neighborhoods fetched: {len(neighborhoods)}")
                    logger.info(f"Sample neighborhood structure: {next(iter(neighborhoods.items())) if neighborhoods else 'None'}")
            except Exception as e:
                logger.error(f"Error during neighborhood traversal: {e}")

        search_res_tuples = []
        if logflag:
            logger.info(f"Processing {len(search_res)} results with {len(neighborhoods)} neighborhoods")

        for r in search_res:
            page_content = r.page_content
            neighborhood = neighborhoods.get(r.id)

            if logflag:
                logger.debug(f"Processing document ID: {r.id}")
                logger.debug(f"Has neighborhood: {neighborhood is not None}")

            text = page_content
            if neighborhood:
                text += "\n------\nRELATED CHUNKS:\n------\n"
                text += f"{neighborhood}\n"

            if logflag:
                logger.info(f"Document: {r.id}, Text: {text}")

            search_res_tuples.append((r.id, text, r.metadata))
            search_res_tuples_summarized = []

        if SUMMARIZER_ENABLED:

            for id, text, metadata in search_res_tuples:
                prompt = self.generate_prompt(query, text)

                res = self.llm.invoke(prompt)
                summarized_text = res.content
                tokens_used = res.usage_metadata        

                if logflag:
                    logger.info(f"Tokens used: {tokens_used}")
                    logger.info(f"SUMMARIZING {id}: {summarized_text}")

                search_res_tuples_summarized.append((id, summarized_text, metadata))

        search_res_tuples = search_res_tuples_summarized

        retrieved_docs: Union[list[TextDoc], list[RetrievalResponseData]] = []
        if isinstance(input, EmbedDoc):
            retrieved_docs = [TextDoc(id=id, text=text) for id, text, _ in search_res_tuples]
            result = SearchedDoc(retrieved_docs=retrieved_docs, initial_query=input.text, top_n=4)

        else:
            retrieved_docs = [
                RetrievalResponseData(id=id, text=text, metadata=metadata) for id, text, metadata in search_res_tuples
            ]

            if isinstance(input, RetrievalRequest):
                result = RetrievalResponse(retrieved_docs=retrieved_docs)

            else:
                input.retrieved_docs = retrieved_docs
                input.documents = [doc.text for doc in retrieved_docs]
                result = input

        statistics_dict["opea_service@retriever_arango"].append_latency(time.time() - start, None)

        if logflag:
            logger.info(retrieved_docs)
            logger.info(result)

        return result

    def generate_prompt(self, query: str, text: str) -> str:
        return f"""You are a helpful, respectful and honest assistant to help the user with summarizartion. \
        Please refer to the search results obtained from the local knowledge base. \
            I've performed vector similarity on query: {query} 
            to retrieve the most relevant documents. Each document has a RELATED CHUNKS section.
            The text that I want you to summarize based on the query above is {text}. 

            Please give the long summary to include all relevant content with statistics including important figures from the RELATED CHUNKS section.
        """
