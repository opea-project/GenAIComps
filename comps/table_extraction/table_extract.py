# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from langsmith import traceable

from comps import TableExtractDoc, opea_microservices, register_microservice
from comps.table_extraction.parser import DocumentParser


@register_microservice(name="opea_service@table_extract", endpoint="/v1/table/extract", host="0.0.0.0", port=6008)
@traceable(run_type="tool")
def table_extraction(input: TableExtractDoc):
    documents = DocumentParser().load(input.path, input.table_strategy)
    return documents


if __name__ == "__main__":
    opea_microservices["opea_service@table_extract"].start()
