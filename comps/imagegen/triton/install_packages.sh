#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -ex

pt_package_name="pytorch_modules-v${PT_VERSION}_${VERSION}_${REVISION}.tgz"
os_string="ubuntu${OS_NUMBER}"
case "${BASE_NAME}" in
    *rhel9*)
        os_string="rhel92"
    ;;
    *amzn2*)
        os_string="amzn2"
    ;;
    *debian*)
        os_string="debian${OS_NUMBER}"
    ;;
    *tencentos*)
        os_string="tencentos31"
    ;;
esac
pt_artifact_path="https://${ARTIFACTORY_URL}/artifactory/gaudi-pt-modules/${VERSION}/${REVISION}/pytorch/${os_string}"
echo pt_artifact_path
tmp_path=$(mktemp --directory)
wget --no-verbose "${pt_artifact_path}/${pt_package_name}"
tar -xf "${pt_package_name}" -C "${tmp_path}"/.
pushd "${tmp_path}"
./install.sh $VERSION $REVISION
popd
# cleanup
rm -rf "${tmp_path}" "${pt_package_name}"
