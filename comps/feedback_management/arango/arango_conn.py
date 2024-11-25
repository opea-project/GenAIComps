# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from arango import ArangoClient as PythonArangoClient
from arango.database import StandardDatabase
from config import ARANGO_HOST, ARANGO_PASSWORD, ARANGO_PORT, ARANGO_USERNAME, DB_NAME, PROTOCOL


class ArangoClient:
    conn_url = f"{PROTOCOL}://{ARANGO_HOST}:{ARANGO_PORT}/"

    @staticmethod
    def get_db_client() -> StandardDatabase:
        try:
            # Create client
            client = PythonArangoClient(hosts=ArangoClient.conn_url)

            # First connect to _system database
            sys_db = client.db("_system", username=ARANGO_USERNAME, password=ARANGO_PASSWORD, verify=True)

            # Create target database if it doesn't exist
            if not sys_db.has_database(DB_NAME):
                sys_db.create_database(DB_NAME)

            # Now connect to the target database
            db = client.db(DB_NAME, username=ARANGO_USERNAME, password=ARANGO_PASSWORD, verify=True)

            return db

        except Exception as e:
            print(e)
            raise e
