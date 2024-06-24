#
# Copyright 2023 The LLM-on-Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import glob
import importlib

from llm_on_ray.common.logging import logger


def import_all_modules(basedir, prefix=None):
    all_py_files = glob.glob(basedir + "/*.py")
    modules = [os.path.basename(f) for f in all_py_files]

    for module in modules:
        if not module.startswith("_"):
            module = module.rstrip(".py")
            if prefix is None:
                module_name = module
            else:
                module_name = f"{prefix}.{module}"
            try:
                importlib.import_module(module_name)
            except Exception:
                logger.warning(f"import {module_name} error", exc_info=True)
