import os
from typing import Any, Union

import openai
from arango import ArangoClient
from arango.database import StandardDatabase
from langchain_arangodb import ArangoVector
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType, EmbedDoc
from comps.cores.proto.api_protocol import RetrievalRequestArangoDB, RetrievalRequest, ChatCompletionRequest

from .config import (
    ARANGO_DB_NAME,
    ARANGO_DISTANCE_STRATEGY,
    ARANGO_GRAPH_NAME,
    ARANGO_NUM_CENTROIDS,
    ARANGO_PASSWORD,
    ARANGO_SEARCH_START,
    ARANGO_TRAVERSAL_ENABLED,
    ARANGO_TRAVERSAL_MAX_DEPTH,
    ARANGO_TRAVERSAL_MAX_RETURNED,
    ARANGO_TRAVERSAL_QUERY,
    ARANGO_TRAVERSAL_SCORE_THRESHOLD,
    ARANGO_URL,
    ARANGO_USE_APPROX_SEARCH,
    ARANGO_USERNAME,
    HUGGINGFACEHUB_API_TOKEN,
    OPENAI_API_KEY,
    OPENAI_CHAT_ENABLED,
    OPENAI_CHAT_MAX_TOKENS,
    OPENAI_CHAT_MODEL,
    OPENAI_CHAT_TEMPERATURE,
    OPENAI_EMBED_ENABLED,
    OPENAI_EMBED_MODEL,
    SUMMARIZER_ENABLED,
    TEI_EMBED_MODEL,
    TEI_EMBEDDING_ENDPOINT,
    VLLM_API_KEY,
    VLLM_ENDPOINT,
    VLLM_MAX_NEW_TOKENS,
    VLLM_MODEL_ID,
    VLLM_TEMPERATURE,
    VLLM_TIMEOUT,
    VLLM_TOP_P,
)

logger = CustomLogger("OPEA_RETRIEVER_ARANGODB")
logflag = os.getenv("LOGFLAG", False)

ARANGO_TEXT_FIELD = "text"
ARANGO_EMBEDDING_FIELD = "embedding"


@OpeaComponentRegistry.register("OPEA_RETRIEVER_ARANGODB")
class OpeaArangoRetriever(OpeaComponent):
    """A specialized retriever component derived from OpeaComponent for ArangoDB retriever services.

    Attributes:
        client (ArangoDB): An instance of the ArangoDB client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RETRIEVER.name.lower(), description, config)

        self._initialize_client()

        if SUMMARIZER_ENABLED:
            self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the language model for summarization if enabled."""
        if OPENAI_API_KEY and OPENAI_CHAT_ENABLED:
            if logflag:
                logger.info("OpenAI API Key is set. Verifying its validity...")

            openai.api_key = OPENAI_API_KEY

            try:
                openai.models.list()
                if logflag:
                    logger.info("OpenAI API Key is valid.")
                self.llm = ChatOpenAI(
                    temperature=OPENAI_CHAT_TEMPERATURE, model=OPENAI_CHAT_MODEL, max_tokens=OPENAI_CHAT_MAX_TOKENS
                )
            except openai.error.AuthenticationError:
                if logflag:
                    logger.info("OpenAI API Key is invalid.")
            except Exception as e:
                if logflag:
                    logger.info(f"An error occurred while verifying the API Key: {e}")

        elif VLLM_ENDPOINT:
            self.llm = ChatOpenAI(
                openai_api_key=VLLM_API_KEY,
                openai_api_base=f"{VLLM_ENDPOINT}/v1",
                model=VLLM_MODEL_ID,
                temperature=VLLM_TEMPERATURE,
                max_tokens=VLLM_MAX_NEW_TOKENS,
                top_p=VLLM_TOP_P,
                timeout=VLLM_TIMEOUT,
            )
        else:
            raise HTTPException(status_code=400, detail="No LLM environment variables are set, cannot generate graphs.")

    def _initialize_client(self):
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
        db: StandardDatabase,
        keys: list[str],
        graph_name: str,
        search_start: str,
        query_embedding: list[float],
        collection_name: str,
        traversal_max_depth: int,
        traversal_max_returned: int,
        traversal_score_threshold: float,
        traversal_query: str,
        distance_strategy: str,
    ) -> dict[str, Any]:
        """Fetch the neighborhoods of matched documents from an ArangoDB graph.
        This method retrieves neighborhoods of documents based on a specified graph traversal
        strategy, distance scoring, and other parameters. It supports different starting points
        for the traversal, such as "chunk", "edge", or "node".

        If `traversal_query` is provided, it will override the default traversal behavior.

        Args:
            db (StandardDatabase): The ArangoDB database instance.
            keys (list[str]): A list of document keys to search for.
            graph_name (str): The name of the graph to traverse.
            search_start (str): The starting point for the traversal. Options are "chunk", "edge", or "node".
            query_embedding (list[float]): The embedding vector used for similarity scoring.
            collection_name (str): The name of the collection containing the documents.
            traversal_max_depth (int): The maximum depth for the graph traversal.
            traversal_max_returned (int): The maximum number of results to return per traversal.
            traversal_score_threshold (float): The minimum score threshold for including results.
            traversal_query (str): A custom traversal query to override the default behavior.
            distance_strategy (str): The distance scoring strategy. Options are "COSINE" or "EUCLIDEAN_DISTANCE".
        Returns:
            dict[str, Any]: A dictionary where keys are document keys and values are their neighborhoods.
        Raises:
            HTTPException: If an invalid distance strategy is provided.
        Notes:
            - The function dynamically constructs an AQL query based on the input parameters.
            - If `logflag` is enabled, the constructed query and bind variables are logged.
        """

        if traversal_max_depth < 1:
            traversal_max_depth = 1

        if traversal_max_returned < 1:
            traversal_max_returned = 1

        score_func = ""
        sort_order = ""

        if distance_strategy == "COSINE":
            score_func = "COSINE_SIMILARITY"
            sort_order = "DESC"
        elif distance_strategy == "EUCLIDEAN_DISTANCE":
            score_func = "L2_DISTANCE"
            sort_order = "ASC"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid distance strategy: {distance_strategy}. Expected 'COSINE' or 'EUCLIDEAN_DISTANCE'.",
            )

        sub_query = ""
        neighborhoods = {}

        bind_vars = {
            "@collection": collection_name,
            "keys": keys,
        }

        if traversal_query:
            sub_query = traversal_query.format(
                graph_name=graph_name,
                traversal_max_depth=traversal_max_depth,
                traversal_max_returned=traversal_max_returned,
                traversal_score_threshold=traversal_score_threshold,
                ARANGO_EMBEDDING_FIELD=ARANGO_EMBEDDING_FIELD,
                ARANGO_TEXT_FIELD=ARANGO_TEXT_FIELD,
            )

            if "@query_embedding" in sub_query:
                bind_vars["query_embedding"] = query_embedding

        elif search_start == "chunk":
            bind_vars["query_embedding"] = query_embedding

            sub_query = f"""
                FOR node IN 1..1 INBOUND doc {graph_name}_HAS_SOURCE
                    FOR node2, edge IN 1..{traversal_max_depth} ANY node {graph_name}_LINKS_TO
                        LET score = {score_func}(edge.{ARANGO_EMBEDDING_FIELD}, @query_embedding)
                        SORT score {sort_order}
                        LIMIT {traversal_max_returned}
                        FILTER score >= {traversal_score_threshold}
                        RETURN edge.{ARANGO_TEXT_FIELD}
            """

        elif search_start == "edge":
            sub_query = f"""
                FOR chunk IN {graph_name}_SOURCE
                    FILTER chunk._key == doc.source_id
                    LIMIT 1
                    RETURN chunk.{ARANGO_TEXT_FIELD}
            """

        elif search_start == "node":
            bind_vars["query_embedding"] = query_embedding

            sub_query = f"""
                FOR node, edge IN 1..{traversal_max_depth} ANY doc {graph_name}_LINKS_TO
                    LET score = {score_func}(edge.{ARANGO_EMBEDDING_FIELD}, @query_embedding)
                    SORT score {sort_order}
                    LIMIT {traversal_max_returned}
                    FILTER score >= {traversal_score_threshold}

                    FOR chunk IN {graph_name}_SOURCE
                        FILTER chunk._key == edge.source_id
                        LIMIT 1
                        RETURN {{[edge.{ARANGO_TEXT_FIELD}]: chunk.{ARANGO_TEXT_FIELD}}}
            """

        query = f"""
            FOR doc IN @@collection
                FILTER doc._key IN @keys

                LET neighborhood = (
                    {sub_query}
                )

                RETURN {{[doc._key]: neighborhood}}
        """

        if logflag:
            logger.info(f"Executing query: {query}")
            logger.info(f"Bind variables: {bind_vars.keys()}")

        cursor = db.aql.execute(query, bind_vars=bind_vars)

        for doc in cursor:
            neighborhoods.update(doc)

        return neighborhoods

    def generate_summarization_prompt(self, query: str, text: str) -> str:
        """Generate a summarization prompt based on the provided query and text.
        This method creates a structured prompt to summarize a document retrieved 
        through vector similarity matching. The summarization is guided by the 
        provided query and optionally leverages a 'RELATED INFORMATION' section 
        within the document to enhance relevance.

        Args:
            query (str): The query string used as the foundation for the summary.
            text (str): The document text to be summarized.
        Returns:
            str: A formatted prompt string instructing how to summarize the document.
        """

        return f"""
            I've performed vector similarity on the following
            query to retrieve most relevant documents: '{query}' 

            Each Document retrieved may have a 'RELATED INFORMATION' section.

            Summarize the Document below using the query as the foundation to your summary.

            Discard any unrelated information that is not relevant to the query.

            If the Document has a 'RELATED INFORMATION' section, use it to help you summarize the Document.

            The document is as follows:

            ------
            {text}
            ------

            Provide a summary to include all content relevant to the query, using the RELATED INFORMATION section (if provided) as needed.

            Your summary:
        """

    async def invoke(self, input: Union[ChatCompletionRequest, RetrievalRequest, RetrievalRequestArangoDB, EmbedDoc]) -> list:
        """Process the retrieval request and return relevant documents."""
        if logflag:
            logger.info(input)

        #################
        # Process Input #
        #################

        query = getattr(input, "input") or getattr(input, "text") 
        if not query:
            if logflag:
                logger.error("Query is empty.")

            return []

        embedding = input.embedding if isinstance(input.embedding, list) else None
        graph_name = getattr(input, "graph_name", ARANGO_GRAPH_NAME)
        search_start = getattr(input, "search_start", ARANGO_SEARCH_START)
        enable_traversal = getattr(input, "enable_traversal", ARANGO_TRAVERSAL_ENABLED)
        enable_summarizer = getattr(input, "enable_summarizer", SUMMARIZER_ENABLED)
        distance_strategy = getattr(input, "distance_strategy", ARANGO_DISTANCE_STRATEGY)
        use_approx_search = getattr(input, "use_approx_search", ARANGO_USE_APPROX_SEARCH)
        num_centroids = getattr(input, "num_centroids", ARANGO_NUM_CENTROIDS)
        traversal_max_depth = getattr(input, "traversal_max_depth", ARANGO_TRAVERSAL_MAX_DEPTH)
        traversal_max_returned = getattr(input, "traversal_max_returned", ARANGO_TRAVERSAL_MAX_RETURNED)
        traversal_score_threshold = getattr(input, "traversal_score_threshold", ARANGO_TRAVERSAL_SCORE_THRESHOLD)
        traversal_query = getattr(input, "traversal_query", ARANGO_TRAVERSAL_QUERY)

        if search_start == "node":
            collection_name = f"{graph_name}_ENTITY"
        elif search_start == "edge":
            collection_name = f"{graph_name}_LINKS_TO"
        elif search_start == "chunk":
            collection_name = f"{graph_name}_SOURCE"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid search_start value: {search_start}. Expected 'node', 'edge', or 'chunk'.",
            )

        if logflag:
            logger.info(f"Graph name: {graph_name}, Start Collection name: {collection_name}")

        #################
        # Validate Data #
        #################

        if not self.db.has_graph(graph_name):
            if logflag:
                graph_names = [g["name"] for g in self.db.graphs()]
                logger.error(f"Graph '{graph_name}' does not exist in ArangoDB. Graphs: {graph_names}")
            return []

        v_col_exists = self.db.graph(graph_name).has_vertex_collection(collection_name)
        e_col_exists = self.db.graph(graph_name).has_edge_collection(collection_name)

        if not (v_col_exists or e_col_exists):
            if logflag:
                collection_names = self.db.graph(graph_name).vertex_collections()
                m = f"Collection '{collection_name}' does not exist in graph '{graph_name}'. Collections: {collection_names}"
                logger.error(m)
            return []

        collection = self.db.collection(collection_name)
        collection_count = collection.count()

        if collection_count == 0:
            if logflag:
                logger.error(f"Collection '{collection_name}' is empty.")
            return []

        if collection_count < num_centroids:
            if logflag:
                m = f"Collection '{collection_name}' has fewer documents ({collection_count}) than the number of centroids ({num_centroids}). Please adjust the number of centroids."
                logger.error(m)
            return []

        ################################
        # Retrieve Embedding Dimension #
        ################################

        random_doc = collection.random()
        random_doc_id = random_doc["_id"]
        embedding = random_doc.get(ARANGO_EMBEDDING_FIELD)

        if not embedding:
            if logflag:
                logger.error(f"Document '{random_doc_id}' is missing field '{ARANGO_EMBEDDING_FIELD}'.")
            return []

        if not isinstance(embedding, list):
            if logflag:
                logger.error(f"Document '{random_doc_id}' has a non-list embedding field, found {type(embedding)}.")
            return []

        dimension = len(embedding)

        if dimension == 0:
            if logflag:
                logger.error(f"Document '{random_doc_id}' has an empty embedding field.")
            return []

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
            collection_name=collection_name,
            embedding_field=ARANGO_EMBEDDING_FIELD,
            text_field=ARANGO_TEXT_FIELD,
            distance_strategy=distance_strategy,
            num_centroids=num_centroids,
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
                    use_approx=use_approx_search,
                )
                search_res = [doc for doc, _ in docs_and_similarities]
            elif input.search_type == "mmr":
                search_res = await vector_db.amax_marginal_relevance_search(
                    query=query,
                    embedding=embedding,
                    k=input.k,
                    fetch_k=input.fetch_k,
                    lambda_mult=input.lambda_mult,
                    use_approx=use_approx_search,
                )
            else:
                search_res = await vector_db.asimilarity_search(
                    query=query,
                    embedding=embedding,
                    k=input.k,
                    use_approx=use_approx_search,
                )
        except Exception as e:
            if logflag:
                logger.error(f"Error during similarity search: {e}")
            return []

        if not search_res:
            if logflag:
                logger.info("No documents found.")
            return []

        if logflag:
            logger.info(f"Found {len(search_res)} documents.")

        ########################################
        # Traverse Source Documents (optional) #
        ########################################

        if enable_traversal:
            keys = [r.id for r in search_res]

            neighborhoods = self.fetch_neighborhoods(
                db=vector_db.db,
                keys=keys,
                graph_name=graph_name,
                search_start=search_start,
                query_embedding=embedding,
                collection_name=collection_name,
                traversal_max_depth=traversal_max_depth,
                traversal_max_returned=traversal_max_returned,
                traversal_score_threshold=traversal_score_threshold,
                traversal_query=traversal_query,
                distance_strategy=distance_strategy,
            )

            for r in search_res:
                neighborhood = neighborhoods.get(r.id)
                if neighborhood:
                    r.page_content += "\n------\nRELATED INFORMATION:\n------\n"
                    r.page_content += str(neighborhood)

            if logflag:
                logger.info(f"Added neighborhoods to {len(search_res)} documents.")

        ################################
        # Summarize Results (optional) #
        ################################

        if enable_summarizer:
            for r in search_res:
                prompt = self.generate_summarization_prompt(query, r.page_content)
                res = self.llm.invoke(prompt)
                summarized_text = res.content

                if logflag:
                    logger.info(f"Summarized {r.id}")

                r.page_content = summarized_text

        return search_res
