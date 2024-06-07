# Safety Check Microservice

# 🚀1. Start Microservice with Python（Option 1）

## 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

## 1.2 Start Safety Check Microservice with Python Script

Start safety check microservice with below command.

```bash
cd /your_project_path/GenAIComps/
cp comps/security/safety_checker.py .
python safety_checker.py
```

# 🚀2. Start Microservice with Docker (Option 2)

## 2.1 Build Docker Image

```bash
cd /your_project_path/GenAIComps
docker build --no-cache -t opea/security:latest -f comps/security/Dockerfile .
```

## 2.2 Run Docker with CLI

```bash
docker run -it --name="/security-server" -p 6008:6008 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/security:latest
```

# 🚀3. Consume Microservice

Once safety check microservice is started, user can use below command to invoke the microservice.

You can prepare your default sensitive wordset in `dict.py`, which defines a a two-dimensional array with n rows and two columns, and each one-dimensional array is a list. See [dict.py](https://github.com/opea-project/GenAIComps/tree/main/comps/security/dict.py) for an example.

You can also prepare your sensitive word dictionary in `dict.txt`, where each line is a string. See [dict.txt](https://github.com/opea-project/GenAIComps/tree/main/comps/security/dict.txt) for an example.

```bash
curl http://${your_ip}:6008/v1/safety/check \
    -H "Content-Type: application/json"   \
    -d '{"text":"bomb the interchange","path":"/your_project_path/GenAIComps/comps/security"}'
```
