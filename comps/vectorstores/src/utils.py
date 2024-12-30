# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import urllib.parse


def encode_filename(filename):
    return urllib.parse.quote(filename, safe="")


def decode_filename(encoded_filename):
    return urllib.parse.unquote(encoded_filename)


def _format_file_dict(file_name):
    file_dict = {
        "name": decode_filename(file_name),
        "id": decode_filename(file_name),
        "type": "File",
        "parent": "",
    }
    return file_dict


def format_search_results(response, file_list: list):
    for i in range(1, len(response), 2):
        file_name = response[i].decode()[5:]
        file_list.append(_format_file_dict(file_name))
    return file_list


def format_search_results_from_list(file_list: list):
    result_list = []
    for file_name in file_list:
        result_list.append(_format_file_dict(file_name))
    return result_list
