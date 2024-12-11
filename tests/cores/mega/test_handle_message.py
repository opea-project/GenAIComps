# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
sys.path.append("/lkk/move_gateway/GenAIComps/")

import json
import unittest
from typing import Union

from comps.cores.mega.utils import handle_message

class TestHandleMessage(unittest.IsolatedAsyncioTestCase):

    def test_handle_message(self):
        messages = [
            {"role": "user", "content": "opea project! "},
        ]
        print(handle_message(messages))
        prompt = handle_message(messages)
        print(prompt)
        self.assertEqual(prompt, "user: opea project! \n")

    def test_handle_message_with_system_prompt(self):
        messages = [
            {"role": "system", "content": "System Prompt"},
            {"role": "user", "content": "opea project! "},
        ]
        prompt = handle_message(messages)
        print(prompt)
        self.assertEqual(prompt, "System Prompt\nuser: opea project! \n")

    def test_handle_message_with_image(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello, "},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    },
                ],
            },
        ]
        prompt, image = handle_message(messages)
        print(prompt)


if __name__ == "__main__":
    # unittest.main()
    t = TestHandleMessage()
    t.test_handle_message_with_image()
