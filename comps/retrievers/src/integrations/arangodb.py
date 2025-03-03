import os
import time
from typing import Any, Union

import openai
from arango import ArangoClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.vectorstores.arangodb_vector import ArangoVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from comps import CustomLogger, EmbedDoc, OpeaComponent, OpeaComponentRegistry, SearchedDoc, ServiceType
from comps.cores.proto.api_protocol import ChatCompletionRequest, RetrievalRequest, RetrievalResponse

from .config import (
    ARANGO_DB_NAME,
    ARANGO_DISTANCE_STRATEGY,
    ARANGO_EMBEDDING_FIELD,
    ARANGO_GRAPH_NAME,
    ARANGO_NUM_CENTROIDS,
    ARANGO_PASSWORD,
    ARANGO_TEXT_FIELD,
    ARANGO_TRAVERSAL_ENABLED,
    ARANGO_TRAVERSAL_MAX_DEPTH,
    ARANGO_TRAVERSAL_MAX_RETURNED,
    ARANGO_URL,
    ARANGO_USE_APPROX_SEARCH,
    ARANGO_USERNAME,
    HUGGINGFACEHUB_API_TOKEN,
    OPENAI_API_KEY,
    OPENAI_CHAT_ENABLED,
    OPENAI_CHAT_MODEL,
    OPENAI_CHAT_TEMPERATURE,
    OPENAI_EMBED_ENABLED,
    OPENAI_EMBED_MODEL,
    SUMMARIZER_ENABLED,
    TEI_EMBED_MODEL,
    TEI_EMBEDDING_ENDPOINT,
    VLLM_ENDPOINT,
    VLLM_MAX_NEW_TOKENS,
    VLLM_MODEL_ID,
    VLLM_TEMPERATURE,
    VLLM_TIMEOUT,
    VLLM_TOP_P,
)

logger = CustomLogger("retriever_arango")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("OPEA_RETRIEVER_ARANGO")
class OpeaArangoRetriever(OpeaComponent):
    """A specialized retriever component derived from OpeaComponent for ArangoDB retriever services.

    Attributes:
        client (ArangoDB): An instance of the ArangoDB client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RETRIEVER.name.lower(), description, config)

        self.initialize_arangodb()

        if SUMMARIZER_ENABLED:
            self.initialize_llm()

    def initialize_llm(self):
        """Initialize the language model for summarization if enabled."""
        if OPENAI_API_KEY and OPENAI_CHAT_ENABLED:
            if logflag:
                logger.info("OpenAI API Key is set. Verifying its validity...")

            openai.api_key = OPENAI_API_KEY

            try:
                openai.models.list()
                if logflag:
                    logger.info("OpenAI API Key is valid.")
                self.llm = ChatOpenAI(temperature=OPENAI_CHAT_TEMPERATURE, model=OPENAI_CHAT_MODEL, max_tokens=512)
            except openai.error.AuthenticationError:
                if logflag:
                    logger.info("OpenAI API Key is invalid.")
            except Exception as e:
                if logflag:
                    logger.info(f"An error occurred while verifying the API Key: {e}")

        elif VLLM_ENDPOINT:
            self.llm = ChatOpenAI(
                openai_api_key="EMPTY",
                openai_api_base=f"{VLLM_ENDPOINT}/v1",
                model=VLLM_MODEL_ID,
                temperature=VLLM_TEMPERATURE,
                max_tokens=VLLM_MAX_NEW_TOKENS,
                top_p=VLLM_TOP_P,
                timeout=VLLM_TIMEOUT,
            )
        else:
            raise ValueError("No LLM text generation environment variables are set, cannot summarize search results.")

    def initialize_arangodb(self):
        """Initialize the ArangoDB connection."""
        self.client = ArangoClient(hosts=ARANGO_URL)
        sys_db = self.client.db(name="_system", username=ARANGO_USERNAME, password=ARANGO_PASSWORD, verify=True)

        if not sys_db.has_database(ARANGO_DB_NAME):
            sys_db.create_database(ARANGO_DB_NAME)

        self.db = self.client.db(name=ARANGO_DB_NAME, username=ARANGO_USERNAME, password=ARANGO_PASSWORD, verify=True)
        if logflag:
            logger.info(f"Connected to ArangoDB {self.db.version()}.")

    def check_health(self) -> bool:
        """Checks the health of the retriever service."""
        if logflag:
            logger.info("[ check health ] start to check health of ArangoDB")
        try:
            version = self.db.version()
            if logflag:
                logger.info(f"[ check health ] Successfully connected to ArangoDB {version}!")
            return True
        except Exception as e:
            logger.info(f"[ check health ] Failed to connect to ArangoDB: {e}")
            return False

    def fetch_neighborhoods(
        self,
        vector_db: ArangoVector,
        keys: list[str],
        graph_name: str,
        source_collection_name: str,
    ) -> dict[str, Any]:
        """Fetch neighborhoods of source documents"""
        neighborhoods = {}

        if ARANGO_TRAVERSAL_MAX_DEPTH <= 0:
            start_vertex = "v1"
            links_to_query = ""
        else:
            start_vertex = "v2"
            links_to_query = f"FOR v2 IN 1..{ARANGO_TRAVERSAL_MAX_DEPTH} ANY v1 {graph_name}_LINKS_TO OPTIONS {{uniqueEdges: 'path'}}"

        if ARANGO_TRAVERSAL_MAX_RETURNED <= 0:
            limit_query = ""
        else:
            limit_query = f"""
                LET score = COSINE_SIMILARITY(doc.{ARANGO_EMBEDDING_FIELD}, s.{ARANGO_EMBEDDING_FIELD})
                SORT score DESC
                LIMIT {ARANGO_TRAVERSAL_MAX_RETURNED}
            """

        aql = f"""
            FOR doc IN @@collection
                FILTER doc._key IN @keys

                LET source_neighborhood = (
                    FOR v1 IN 1..1 INBOUND doc {graph_name}_HAS_SOURCE
                        {links_to_query}
                            FOR s IN 1..1 OUTBOUND {start_vertex} {graph_name}_HAS_SOURCE
                                FILTER s._key != doc._key
                                {limit_query}
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

        return neighborhoods

    def generate_prompt(self, query: str, text: str) -> str:
        """Generate a prompt for summarization."""
        return f"""
            I've performed vector similarity on the following
            query to retrieve most relevant documents: '{query}' 

            Each retrieved Document may have a 'RELATED CHUNKS' section.

            Please consider summarizing the Document below using the query as the foundation to summarize the text.

            The Document: {text}

            Provide a summary to include all content relevant to the query, using the RELATED CHUNKS section (if provided) as needed.

            Your summary:
        """

    async def invoke(
        self, input: Union[EmbedDoc, RetrievalRequest, ChatCompletionRequest]
    ) -> Union[SearchedDoc, RetrievalResponse, ChatCompletionRequest]:
        """Process the retrieval request and return relevant documents."""
        if logflag:
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

        ########################
        # Fetch the Graph Name #
        ########################

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

        ################################
        # Retrieve Embedding Dimension #
        ################################

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

        if OPENAI_API_KEY and OPENAI_EMBED_MODEL and OPENAI_EMBED_ENABLED:
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

        ######################
        # Compute Similarity #
        ######################

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

        if logflag:
            logger.info(f"Found {len(search_res)} documents.")

        ########################################
        # Traverse Source Documents (optional) #
        ########################################

        if ARANGO_TRAVERSAL_ENABLED:
            neighborhoods = self.fetch_neighborhoods(
                vector_db=vector_db,
                keys=[r.id for r in search_res],
                graph_name=graph_name,
                source_collection_name=source_collection_name,
            )

            for r in search_res:
                neighborhood = neighborhoods.get(r.id)
                if neighborhood:
                    r.page_content += "\n------\nRELATED CHUNKS:\n------\n"
                    r.page_content += str(neighborhood)

            if logflag:
                logger.info(f"Added neighborhoods to {len(search_res)} documents.")

        ################################
        # Summarize Results (optional) #
        ################################

        if SUMMARIZER_ENABLED:
            for r in search_res:
                prompt = self.generate_prompt(query, r.page_content)
                res = self.llm.invoke(prompt)
                summarized_text = res.content

                if logflag:
                    logger.info(f"Summarized {r.id}")

                r.page_content = summarized_text

        return search_res
