# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from comps.cores.proto.docarray import DocList


def process_doc_list(docs: DocList) -> List[str]:
    result = []
    for doc in docs:
        if doc.text is None:
            continue
        if isinstance(doc.text, list):
            result.extend(doc.text)
        else:
            result.append(doc.text)
    return result
