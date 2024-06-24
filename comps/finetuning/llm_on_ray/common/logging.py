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

import logging
import logging.config
import traceback
import functools

__all__ = ["logger", "get_logger"]

use_accelerate_log = False
logger_name = "common"

logging_config = {
    "version": 1,
    "loggers": {
        "root": {"level": "INFO", "handlers": ["consoleHandler"]},
        "common": {
            "level": "INFO",
            "handlers": ["consoleHandler"],
            "qualname": "common",
            "propagate": 0,
        },
    },
    "handlers": {
        "consoleHandler": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standardFormatter",
        },
    },
    "formatters": {
        "standardFormatter": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "",
        }
    },
}

if logging_config is not None:
    try:
        logging.config.dictConfig(logging_config)
    except Exception:
        traceback.print_exc()
        exit(1)

if use_accelerate_log:
    import accelerate

    get_logger = functools.partial(accelerate.logging.get_logger, name=logger_name)
else:
    get_logger = functools.partial(logging.getLogger, name=logger_name)

logger = get_logger()
