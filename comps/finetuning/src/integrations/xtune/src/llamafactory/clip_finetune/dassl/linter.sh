# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

echo "Running isort"
isort -y -sp .
echo "Done"

echo "Running yapf"
yapf -i -r -vv -e build .
echo "Done"

echo "Running flake8"
flake8 .
echo "Done"
