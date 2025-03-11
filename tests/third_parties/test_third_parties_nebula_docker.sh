#!/bin/bash
set -euo pipefail

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"

function check_storage_class() {
    echo "Checking storage class..."
    STORAGE_CLASS_NAME=$(docker system info --format '{{.DriverStatus}}' | grep "Storage Driver" | cut -d':' -f2- | tr -d ' ')
    
    if [ -n "$STORAGE_CLASS_NAME" ]; then
        echo "Using Storage Driver: $STORAGE_CLASS_NAME"
    else
        echo "No Storage Driver found."
        exit 1
    fi
}

function start_service() {
    echo "Starting Nebula Graph service..."
    docker-compose up -d
    
    # Wait for graphd to be ready
    for i in {1..60}; do
        status_code=$(curl -s -o /dev/null -w "%{http_code}" http://graphd:19669/status)
        if [ "$status_code" = "200" ]; then
            break
        fi
        sleep 1
    done
    
    if [ "$status_code" != "200" ]; then
        echo "Failed to start Nebula Graph"
        exit 1
    fi
}

function validate_database() {
    echo "[ test create ] creating space..."
    
    query="CREATE SPACE my_space(partition_num=10, replica_factor=1, vid_type=FIXED_STRING(32)); USE my_space; CREATE TAG person(name string, age int);"
    create_response=$(docker exec -it console nebula-console -addr graphd -port 9669 -u $NEBULA_USER -p $NEBULA_PASSWORD -e "$query" 2>&1)
    
    if [[ $? -eq 0 ]]; then
        echo "[ test create ] create space succeed"
        echo $create_response >> ${LOG_PATH}/nebulagraph_create_space.log
    else
        echo "[ test create ] create space failed"
        echo $create_response >> ${LOG_PATH}/nebulagraph_create_space.log
        exit 1
    fi
    
    sleep 30s
    
    # Test insert data
    echo "[ test insert ] inserting data..."
    query="USE my_space; INSERT VERTEX person(name, age) VALUES 'person1':('Alice', 30); INSERT VERTEX person(name, age) VALUES 'person2':('Bob', 25);"
    insert_response=$(docker exec -it console nebula-console -addr graphd -port 9669 -u $NEBULA_USER -p $NEBULA_PASSWORD -e "$query" 2>&1)
    
    if [[ $? -eq 0 ]]; then
        echo "[ test insert ] insert data succeed"
        echo $insert_response >> ${LOG_PATH}/nebulagraph_insert_data.log
    else
        echo "[ test insert ] insert data failed"
        echo $insert_response >> ${LOG_PATH}/nebulagraph_insert_data.log
        exit 1
    fi
    
    sleep 30s
    
    # Test search data
    echo "[ test search ] searching data..."
    query="USE my_space; MATCH (p:person) RETURN p;"
    search_response=$(docker exec -it console nebula-console -addr graphd -port 9669 -u $NEBULA_USER -p $NEBULA_PASSWORD -e "$query" 2>&1)
    
    if [[ $? -eq 0 ]]; then
        echo "[ test search ] search data succeed"
        echo $search_response >> ${LOG_PATH}/nebulagraph_search_data.log
    else
        echo "[ test search ] search data failed"
        echo $search_response >> ${LOG_PATH}/nebulagraph_search_data.log
        exit 1
    fi
}

function stop_service() {
    echo "Stopping Nebula Graph service..."
    docker-compose down
}

function main() {
    check_storage_class
    mkdir -p "$LOG_PATH"
    start_service
    validate_database
    stop_service
}

main "$@"
