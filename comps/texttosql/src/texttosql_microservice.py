# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import sys

from fastapi.exceptions import HTTPException

from comps import CustomLogger, OpeaComponentController, opea_microservices, register_microservice
from comps.texttosql.src.integrations.opea import Input, OpeaTextToSQL

cur_path = pathlib.Path(__file__).parent.resolve()
comps_path = os.path.join(cur_path, "../../../")
sys.path.append(comps_path)

logger = CustomLogger("texttosql")
logflag = os.getenv("LOGFLAG", False)

try:
    # Initialize OpeaComponentController
    controller = OpeaComponentController()

    # Register components
    texttosql_agent = OpeaTextToSQL(
        name="TextToSQL",
        description="TextToSQL Service",
    )

    # Register components with the controller
    controller.register(texttosql_agent)

    # Discover and activate a healthy component
    controller.discover_and_activate()
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")


@register_microservice(
    name="opea_service@texttosql",
    endpoint="/v1/texttosql",
    host="0.0.0.0",
    port=8080,
)
async def execute_agent(input: Input):
    """Execute a SQL query from the input text.

    This function takes an Input object containing the input text and database connection information.
    It uses the execute function from the texttosql module to execute the SQL query and returns the result.

    Args:
        input (Input): An Input object with the input text and database connection information.

    Returns:
        dict: A dictionary with a 'result' key containing the output of the executed SQL query.
    """
    if input.conn_str.test_connection():
        response = await controller.invoke(input)
        # response = "a"
        return {"result": response}
    else:
        raise HTTPException(status_code=500, detail="Failed to connect to PostgreSQL database")


if __name__ == "__main__":
    logger.info("OPEA TextToSQL Microservice is starting...")
    opea_microservices["opea_service@texttosql"].start()
