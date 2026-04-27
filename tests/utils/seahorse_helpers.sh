#!/bin/bash
# Copyright (C) 2026 Dnotitia
# SPDX-License-Identifier: Apache-2.0

require_seahorse_credentials_or_skip() {
    local missing=()

    if [ -z "${SEAHORSE_BASE_URL:-}" ]; then
        missing+=("SEAHORSE_BASE_URL")
    fi
    if [ -z "${SEAHORSE_API_KEY:-}" ]; then
        missing+=("SEAHORSE_API_KEY")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo "[SKIP] Seahorse credentials not configured: ${missing[*]}"
        exit 0
    fi
}

detect_host_ip() {
    local detected_host_ip=""

    detected_host_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [ -z "$detected_host_ip" ] && command -v ipconfig >/dev/null 2>&1; then
        detected_host_ip=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)
    fi
    if [ -z "$detected_host_ip" ]; then
        detected_host_ip="127.0.0.1"
    fi

    export host_ip="$detected_host_ip"
}
