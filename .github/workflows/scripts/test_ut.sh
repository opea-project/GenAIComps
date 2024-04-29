#!/bin/bash

# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

PATTERN='[-a-zA-Z0-9_]*='
for i in "$@"
do
    case $i in
        --test_name=*)
            test_name=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

# setup test ENV
TODO

# run test
ut_log_name=/GenAIComps/.github/workflows/scripts/${test_name}_ut.log
cd /GenAICompos/tests/uts
if [ $test_name = 'mega' ]; then
    echo "run mega test"
    python -m pytest -vs ./mega 2>&1 | tee ${ut_log_name}
else
    echo "run other test"
    python -m pytest -vs ./test_${test_name}*.py 2>&1 | tee ${ut_log_name}
fi

# check test result
if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed' ${ut_log_name}) == 0 ]; then
    echo "Find errors in pytest case, please check the output..."
    echo "Please search for '== FAILURES ==' or '== ERRORS =='"
    exit 1
fi
echo "UT finished successfully! "
