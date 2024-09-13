# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import sys

from pydantic import BaseModel , Field
from typing import (
    Optional,
    Annotated,
)

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from comps.texttosql.langchain.src.texttosql import execute

cur_path = pathlib.Path(__file__).parent.resolve()
comps_path = os.path.join(cur_path, "../../../")
sys.path.append(comps_path)

from comps import CustomLogger, opea_microservices, register_microservice

logger = CustomLogger("comps-texttosql")
logflag = os.getenv("LOGFLAG", False)

class PostgresConnection(BaseModel):
    user: Annotated[str, Field(min_length=1)]
    password: Annotated[str, Field(min_length=1)]
    host: Annotated[str, Field(min_length=1)]
    port: Annotated[
        int, Field(ge=1, le=65535)
    ]  # Default PostgreSQL port with constraints
    database: Annotated[str, Field(min_length=1)]

    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def test_connection(self) -> bool:
        """Test the connection to the PostgreSQL database."""
        connection_string = self.connection_string()
        try:
            engine = create_engine(connection_string)
            with engine.connect() as connection:
                # If the connection is successful, return True
                return True
        except SQLAlchemyError as e:
            print(f"Connection failed: {e}")
            return False


class Input(BaseModel):
    input_text: str
    conn_str: Optional[PostgresConnection] = None
    
@register_microservice(
    name="opea_service@comps-texttosql",
    endpoint="/v1/test-connection",
    host="0.0.0.0",
    port=8080,
)
def test_connection(input: PostgresConnection):
    # Test the database connection
    result = input.test_connection()
    if result:
        return True
    else:
        return False

@register_microservice(
    name="opea_service@comps-texttosql",
    endpoint="/v1/texttosql",
    host="0.0.0.0",
    port=8080,
)
def execute_agent(input: Input):
    # Test the database connection
    logger.info("User Query: {}".format(input.input_text))
    url = input.conn_str.connection_string()
    if input.conn_str.test_connection():
        result = execute(input.input_text, url)
        return {"result": result}



if __name__ == "__main__":
    opea_microservices["opea_service@comps-texttosql"].start()