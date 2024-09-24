#!/usr/bin/env bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

python lvm_tp_server.py &
python lvm_tp.py &
