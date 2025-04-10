# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

# name => statistic dict
statistics_dict = {}


class BaseStatistics:
    """Base class to store in-memory statistics of an entity for measurement in one service."""

    def __init__(
        self,
    ):
        self.response_times = []  # store responses time for all requests
        self.first_token_latencies = []  # store first token latencies for all requests

    def append_latency(self, latency, first_token_latency=None):
        self.response_times.append(latency)
        if first_token_latency:
            self.first_token_latencies.append(first_token_latency)

    def _add_statistics(self, result, stats, suffix):
        "add P50 (median), P99 and average values for 'stats' array to 'result' dict"
        if stats:
            result[f"p50_{suffix}"] = np.percentile(stats, 50)
            result[f"p99_{suffix}"] = np.percentile(stats, 99)
            result[f"average_{suffix}"] = np.average(stats)
        else:
            result[f"p50_{suffix}"] = None
            result[f"p99_{suffix}"] = None
            result[f"average_{suffix}"] = None

    def get_statistics(self):
        "return stats dict with P50, P99 and average values for first token and response timings"
        result = {}
        self._add_statistics(result, self.response_times, "latency")
        self._add_statistics(result, self.first_token_latencies, "latency_first_token")
        return result


def register_statistics(
    names,
):
    def decorator(func):
        for name in names:
            statistics_dict[name] = BaseStatistics()
        return func

    return decorator


def collect_all_statistics():
    results = {}
    if statistics_dict:
        for name, statistic in statistics_dict.items():
            results[name] = statistic.get_statistics()
    return results
