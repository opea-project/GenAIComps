# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from langchain_opea import __all__

EXPECTED_ALL = [
    "OPEALLM",
    "ChatOPEA",
    "OPEAEmbeddings",
    "__version__",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
