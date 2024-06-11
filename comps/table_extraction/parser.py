# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Wrapper for parsing the uploaded user file and then make document indexing."""

import logging
import os
from typing import List

from comps.table_extraction.context_utils import get_chuck_data, load_unstructured_data

logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO
)


class DocumentParser:
    def __init__(self, max_chuck_size=512, min_chuck_size=5, process=True):
        """Wrapper for document parsing."""
        self.max_chuck_size = max_chuck_size
        self.min_chuck_size = min_chuck_size
        self.process = process

    def load(self, input, table_strategy="fast"):
        """The API for loading the file.

        Support single file, batch files, and urls parsing.
        """
        self.table_strategy = table_strategy

        if isinstance(input, str):
            if os.path.isfile(input):
                data_collection = self.parse_document(input)
            elif os.path.isdir(input):
                data_collection = self.batch_parse_document(input)
            else:
                print("Please check your upload file and try again!")
        elif isinstance(input, List):
            try:
                data_collection = self.parse_html(input)
            except:
                logging.error("The given link/str is unavailable. Please try another one!")
        else:
            logging.error("The input format is invalid!")

        return data_collection

    def parse_document(self, input):
        """Parse the uploaded file."""
        if input.endswith("pdf"):
            content, tables = load_unstructured_data(input, self.table_strategy)
            if self.process:
                chuck = get_chuck_data(content, self.max_chuck_size, self.min_chuck_size, input)
            else:
                chuck = [[content.strip(), input]]
            if tables is not None:
                chuck = chuck + tables
        else:
            logging.info("This file {} is ignored. Will support this file format soon.".format(input))
            raise Exception("[Rereieval ERROR] Document format not supported!")
        return chuck

    def batch_parse_document(self, input):
        """Parse the uploaded batch files in the input folder."""
        paragraphs = []
        for dirpath, dirnames, filenames in os.walk(input):
            for filename in filenames:
                if filename.endswith("pdf"):
                    content, tables = load_unstructured_data(os.path.join(dirpath, filename), self.table_strategy)
                    if self.process:
                        chuck = get_chuck_data(content, self.max_chuck_size, self.min_chuck_size, input)
                    else:
                        chuck = [[content.strip(), input]]
                    if tables is not None:
                        chuck = chuck + tables
                    paragraphs += chuck
                else:
                    logging.info("This file {} is ignored. Will support this file format soon.".format(filename))
                    raise Exception("[Rereieval ERROR] Document format not supported!")
        return paragraphs
