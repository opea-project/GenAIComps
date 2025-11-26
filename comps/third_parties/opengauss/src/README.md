# Start openGauss server

## 1. Download openGauss image

```bash
docker pull opengauss/opengauss:7.0.0-RC2.B015
```

## 2. Configure the username, password and dbname

```bash
export GS_PASSWORD=openGauss@123
```

## 3. Run openGauss service

```bash
docker run --name opengauss-db -e GS_PASSWORD=${GS_PASSWORD} -d -p 5432:5432 opengauss/opengauss:7.0.0-RC2.B015
```
