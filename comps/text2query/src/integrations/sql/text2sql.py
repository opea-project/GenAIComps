# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Annotated, Optional

from langchain.agents.agent_types import AgentType
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_huggingface import HuggingFaceEndpoint
from pydantic import BaseModel, Field
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

    async def invoke(self, request: Text2QueryRequest):
        url = request.conn_url
        if not url:
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
