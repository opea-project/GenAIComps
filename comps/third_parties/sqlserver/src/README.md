
# Start SQLServer server


## 1. Download SQLServer image

```bash
docker pull mcr.microsoft.com/mssql/server:2025-latest
```

## 2. Configure the password.

```bash
export MSSQL_SA_PASSWORD='Passw0rd!'
```

**Caution:** Your password should follow the SQL Server default [password policy](https://learn.microsoft.com/en-us/sql/relational-databases/security/password-policy?view=sql-server-ver17). By default, the password must be at least eight characters long and contain characters from three of the following four sets: uppercase letters, lowercase letters, base-10 digits, and symbols. Passwords can be up to 128 characters long. If you don't follow these password requirements, the container can't set up SQL Server, and stops working. You can examine the error log by using the [`docker logs`](https://docs.docker.com/reference/cli/docker/container/logs) command.

## 3. Run SQLServer service

 1. Using docker run command

```bash
docker run -e "ACCEPT_EULA=Y" -e "MSSQL_SA_PASSWORD=$MSSQL_SA_PASSWORD" \
-p 1433:1433 --name sqlserver-db --hostname sql-db \
-d \
mcr.microsoft.com/mssql/server:2025-latest
```

2. Using docker compose command
```bash
cd ../deployment/docker_compose/
docker compose up -d
```