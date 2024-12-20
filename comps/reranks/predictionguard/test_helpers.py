# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identifier: Apache-2.0

import unittest

from comps.cores.proto.docarray import DocList, TextDoc
from comps.reranks.predictionguard.helpers import process_doc_list


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # Test cases setup
        self.simple_case = DocList(
            [TextDoc(text="Hello world"), TextDoc(text="Another document"), TextDoc(text="Third document")]
        )

        self.list_case = DocList(
            [TextDoc(text=["First sentence.", "Second sentence."]), TextDoc(text=["Another paragraph.", "More text."])]
        )

        self.mixed_case = DocList(
            [
                TextDoc(text="Single string document"),
                TextDoc(text=["First part", "Second part"]),
                TextDoc(text="Another single string"),
            ]
        )

        self.none_case = DocList([TextDoc(text="Valid text"), TextDoc(text=[]), TextDoc(text="More text")])

    def test_simple_case(self):
        result = process_doc_list(self.simple_case)
        expected = ["Hello world", "Another document", "Third document"]
        self.assertEqual(result, expected)

    def test_list_case(self):
        result = process_doc_list(self.list_case)
        expected = ["First sentence.", "Second sentence.", "Another paragraph.", "More text."]
        self.assertEqual(result, expected)

    def test_mixed_case(self):
        result = process_doc_list(self.mixed_case)
        expected = ["Single string document", "First part", "Second part", "Another single string"]
        self.assertEqual(result, expected)

    def test_none_case(self):
        result = process_doc_list(self.none_case)
        expected = ["Valid text", "More text"]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
