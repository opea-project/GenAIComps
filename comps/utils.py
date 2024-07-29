# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import uuid


def random_uuid() -> str:
    return str(uuid.uuid4().hex)
