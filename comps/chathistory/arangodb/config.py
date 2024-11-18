# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

# ARANGO configuration
ARANGODB_HOST = os.getenv("ARANGODB_HOST", "localhost")
ARANGODB_PORT = os.getenv("ARANGODB_PORT", 8529)
ARANGODB_USERNAME = os.getenv("ARANGODB_USERNAME", "root")
ARANGODB_PASSWORD = os.getenv("ARANGODB_PASSWORD", "test")
DB_NAME = os.getenv("DB_NAME", "OPEA")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ChatHistory") 