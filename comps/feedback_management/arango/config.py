# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

# ARANGO configuration
ARANGO_HOST = os.getenv("ARANGO_HOST", "localhost")
ARANGO_PORT = os.getenv("ARANGO_PORT", 8529)
ARANGO_USERNAME = os.getenv("ARANGO_USERNAME", "root")
ARANGO_PASSWORD = os.getenv("ARANGO_PASSWORD", "test")
DB_NAME = os.getenv("DB_NAME", "OPEA")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Feedback")