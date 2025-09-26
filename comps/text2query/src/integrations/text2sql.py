# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse

from fastapi.exceptions import HTTPException
from langchain.agents.agent_types import AgentType
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_huggingface import HuggingFaceEndpoint
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import Text2QueryRequest
from comps.text2sql.src.integrations.sql_agent import CustomSQLDatabaseToolkit, custom_create_sql_agent

logger = CustomLogger("comps_text2query_sql")
logflag = os.getenv("LOGFLAG", False)

sql_params = {
    "max_string_length": 3600,
}

generation_params = {
    "max_new_tokens": 1024,
    "top_k": 10,
    "top_p": 0.95,
    "temperature": 0.01,
    "repetition_penalty": 1.03,
    "streaming": True,
}

TGI_LLM_ENDPOINT = os.environ.get("TGI_LLM_ENDPOINT")

llm = HuggingFaceEndpoint(
    endpoint_url=TGI_LLM_ENDPOINT,
    task="text-generation",
    **generation_params,
)


@OpeaComponentRegistry.register("OPEA_TEXT2QUERY_SQL")
class OpeaText2SQL(OpeaComponent):
    """A specialized text to sql component derived from OpeaComponent for interacting with TGI services and Database.

    Attributes:
        client: An instance of the client for text to sql generation and execution.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.TEXT2QUERY.name.lower(), description, config)
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaText2SQL health check failed.")

    async def check_health(self) -> bool:
        """Checks the health of the TGI service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            response = llm.generate(["Hello, how are you?"])
            return True
        except Exception as e:
            return False

    def format_db_url(self, request: Text2QueryRequest) -> str:
        """Format the database connection URL by adding username and password if needed.

        This method takes a database connection URL and ensures it includes authentication
        credentials when provided. If the URL already contains a username, it returns the
        URL as-is. If no username is present in the URL but credentials are provided in
        the request, it constructs a new URL with the username and password embedded.

        Args:
            request (Text2QueryRequest): The request object containing connection details
                including conn_url, conn_username, and conn_password.

        Returns:
            str: A properly formatted database connection URL with credentials if needed.

        Raises:
            ValueError: If no database connection URL is provided in the request.
        """
        url = request.conn_url
        if not url:
            raise ValueError("Database connection URL must be provided in 'conn_url' field of the request.")

        parsed_url = urlparse(url)
        has_username = bool(parsed_url.username)
        if has_username:
            return url
        elif request.conn_username:
            password = request.conn_password if request.conn_password else ""
            new_netloc = f"{request.conn_username}:{password}@{parsed_url.hostname}:{parsed_url.port}"
            updated_parsed_url = parsed_url._replace(netloc=new_netloc)
            updated_jdbc_url = urlunparse(updated_parsed_url)
            return updated_jdbc_url
        else:
            return url

    async def invoke(self, request: Text2QueryRequest):
        url = request.conn_url
        if url:
            url = self.format_db_url(request)
        else:
            raise ValueError("Database connection URL must be provided in 'conn_url' field of the request.")

        """Execute a SQL query using the custom SQL agent.

        Args:
            input (str): The user's input.
            url (str): The URL of the database to connect to.

        Returns:
            dict: The result of the SQL execution.
        """
        db = SQLDatabase.from_uri(url, **sql_params)
        logger.info("Starting Agent")
        agent_executor = custom_create_sql_agent(
            llm=llm,
            verbose=True,
            toolkit=CustomSQLDatabaseToolkit(llm=llm, db=db),
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_executor_kwargs={"return_intermediate_steps": True},
        )

        result = await agent_executor.ainvoke(request)

        query = []
        for log, _ in result["intermediate_steps"]:
            if log.tool == "sql_db_query":
                query.append(log.tool_input)
        result["sql"] = query[0].replace("Observation", "")
        return {"result": result}

    async def db_connection_check(self, request: Text2QueryRequest):
        """Check the connection to the database.

        This function takes a Text2QueryRequest object containing the database connection information.
        It attempts to connect to the database using the provided connection URL and credentials.

        Args:
            request (Text2QueryRequest): A Text2QueryRequest object with the database connection information.
        Returns:
            dict: A dictionary with a 'status' key indicating whether the connection was successful or failed.
        """
        url = request.conn_url
        if url:
            url = self.format_db_url(request)
        else:
            raise ValueError("Database connection URL must be provided in 'conn_url' field of the request.")

        try:
            engine = create_engine(url)
            with engine.connect() as _:
                # If the connection is successful, return True
                return {"status": "Connection successful"}
        except SQLAlchemyError as e:
            logger.error(f"Connection failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to connect to database: {url}")
