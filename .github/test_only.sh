# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#This is for test only
docker images
docker stop $(docker ps -a -q) # this is the dangerous cmd
docker stop # this is harmless
docker ps -a # this is harmless
docker ps -q # this is harmless
sudo rm -fr  # this is the dangerous cmd
rm -fr # this is harmless
