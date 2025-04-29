# Start MariaDB Server

**MariaDB Vector** was introduced starting with server version 11.7  
For more details please see the [official documentation](https://mariadb.com/kb/en/vectors/).


## 1. Configure the server

```bash
export MARIADB_CONTAINER_IMAGE="mariadb:latest"
export MARIADB_USER=dbuser
export MARIADB_PASSWORD=password
export MARIADB_DATABASE=vectordb
```

## 2. Run MariaDB Server

```bash
docker run --name mariadb-server -e MARIADB_USER=${MARIADB_USER} -e MARIADB_RANDOM_ROOT_PASSWORD=1 -e MARIADB_DATABASE=${MARIADB_DATABASE} -e MARIADB_PASSWORD=${MARIADB_PASSWORD} -d -p 3306:3306 ${MARIADB_CONTAINER_IMAGE}
```
