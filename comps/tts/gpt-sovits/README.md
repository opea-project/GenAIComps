
```bash
docker build -t opea/gpt-sovits:latest --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -f comps/tts/gpt-sovits/Dockerfile .

docker run  -itd -p 9880:9880 -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/gpt-sovits:latest

```