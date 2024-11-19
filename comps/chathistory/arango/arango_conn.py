# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# import motor.motor_asyncio as motor
from arango import ArangoClient as PythonArangoClient
from config import DB_NAME, ARANGODB_HOST, ARANGODB_PORT, ARANGODB_PASSWORD, ARANGODB_USERNAME

class ArangoClient:
    conn_url = f"http://{ARANGODB_HOST}:{ARANGODB_PORT}/"

    @staticmethod
    def get_db_client():
        try:
            # Create client
            print(f"Connecting to database: {ArangoClient.conn_url}, username: {ARANGODB_USERNAME}, password: {ARANGODB_PASSWORD}, db: {DB_NAME}")
            client = PythonArangoClient(hosts=ArangoClient.conn_url)
            
            # First connect to _system database
            sys_db = client.db(
                '_system',
                username=ARANGODB_USERNAME,
                password=ARANGODB_PASSWORD,
                verify=True
            )
            print("Connected to _system database")
            
            # Create target database if it doesn't exist
            if not sys_db.has_database(DB_NAME):
                sys_db.create_database(DB_NAME)
                print(f"Created database {DB_NAME}")
            
            # Now connect to the target database
            db = client.db(
                DB_NAME,
                username=ARANGODB_USERNAME,
                password=ARANGODB_PASSWORD,
                verify=True    
            )
            print(f"Connected to database {DB_NAME}")
            
            return db

        except Exception as e:
            print(f"Failed to connect to database: {str(e)}, url: {ArangoClient.conn_url}, username: {ARANGODB_USERNAME}, password: {ARANGODB_PASSWORD}, db: {DB_NAME}")
            raise e
