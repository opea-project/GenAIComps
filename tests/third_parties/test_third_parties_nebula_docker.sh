# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash
set +e
set -x

# Constants
WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
MAX_ATTEMPTS=30
SLEEP_TIME=2

# Function to start NebulaGraph services
start_service() {
    echo "Starting NebulaGraph services..."
    cd $WORKPATH/comps/third_parties/nebula/deployment/docker_compose/
    docker compose up -d


    echo "Waiting for services to become healthy..."
    for ((i=1; i<=MAX_ATTEMPTS; i++)); do
        HEALTHY_COUNT=$(docker compose ps --filter "status=healthy" | wc -l)
        TOTAL_COUNT=$(docker compose ps | wc -l)

        echo "Attempt $i/$MAX_ATTEMPTS: $HEALTHY_COUNT out of $TOTAL_COUNT services are healthy"

        if [ $HEALTHY_COUNT -eq $TOTAL_COUNT ]; then
            echo "All services are healthy!"
            break
        fi

        sleep $SLEEP_TIME
        echo "Still waiting... ($i/$MAX_ATTEMPTS attempts)"
    done

    if [ $i -eq $MAX_ATTEMPTS ]; then
        echo "Services did not become healthy within timeout period"
        echo "Current service statuses:"
        docker compose ps
        exit 1
    fi
}

# Function to validate database operations
validate_database() {
    GRAPHD_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $(docker compose ps -q graphd))

    if [ -z "$GRAPHD_IP" ]; then
        echo "Could not determine graphd IP address"
        exit 1
    fi

    echo "Using graphd IP: $GRAPHD_IP"

    echo "[ test create ] creating space..."
    query="CREATE SPACE my_space(partition_num=10, replica_factor=1, vid_type=FIXED_STRING(32)); USE my_space; CREATE TAG person(name string, age int);"

    create_response=$(docker compose exec console nebula-console -addr "$GRAPHD_IP" -port 9669 -u root -p nebula -e "$query" 2>&1)

    if [[ $? -eq 0 ]]; then
        echo "[ test create ] create space succeed"
    else
        echo "[ test create ] create space failed"
        exit 1
    fi

    sleep $SLEEP_TIME

    echo "[ test insert ] inserting data..."
    query="USE my_space; INSERT VERTEX person(name, age) VALUES 'person1':('Alice', 30); INSERT VERTEX person(name, age) VALUES 'person2':('Bob', 25);"

    insert_response=$(docker compose exec console nebula-console -addr "$GRAPHD_IP" -port 9669 -u root -p nebula -e "$query" 2>&1)

    if [[ $? -eq 0 ]]; then
        echo "[ test insert ] insert data succeed"
    else
        echo "[ test insert ] insert data failed"
        exit 1
    fi

    sleep $SLEEP_TIME

    echo "[ test search ] searching data..."
    query="USE my_space; MATCH (p:person) RETURN p;"

    search_response=$(docker compose exec console nebula-console -addr "$GRAPHD_IP" -port 9669 -u root -p nebula -e "$query" 2>&1)

    if [[ $? -eq 0 ]]; then
        echo "[ test search ] search data succeed"
    else
        echo "[ test search ] search data failed"
        exit 1
    fi
}

# Function to stop services
stop_service() {
    echo "Stopping NebulaGraph services..."
    docker compose down --remove-orphans
}

# Main function
main() {
    stop_service
    start_service
    validate_database
    stop_service
}

# Run main function
main "$@"
