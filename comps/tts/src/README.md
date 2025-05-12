# TTS Microservice

TTS (Text-To-Speech) microservice helps users convert text to speech. When building a talking bot with LLM, users might need an LLM generated answer in audio format. This microservice is built for that conversion stage.

## 1.2 Start SpeechT5 Service/Test

- Xeon CPU

```bash
cd comps/third_parties/speecht5/src
nohup python speecht5_server.py --device=cpu &
curl http://localhost:7055/v1/tts -XPOST -d '{"text": "Who are you?"}' -H 'Content-Type: application/json'
```

- Gaudi2 HPU

```bash
pip install optimum[habana]

cd comps/third_parties/speecht5/src
nohup python speecht5_server.py --device=hpu &
curl http://localhost:7055/v1/tts -XPOST -d '{"text": "Who are you?"}' -H 'Content-Type: application/json'
```

## 1.3 Start TTS Service/Test

```bash
python opea_tts_microservice.py

curl http://localhost:9088/v1/audio/speech -XPOST -d '{"input": "Who are you?"}' -H 'Content-Type: application/json' --output speech.mp3
```

## 🚀2. Start Microservice with Docker (Option 2)

Alternatively, you can start the TTS microservice with Docker.

### 2.1 Build Images

#### 2.1.1 SpeechT5 Server Image

- Xeon CPU

```bash
cd ../../../
docker build -t opea/speecht5:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/third_parties/speecht5/src/Dockerfile .
```

- Gaudi2 HPU

```bash
cd ../../../
docker build -t opea/speecht5-gaudi:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/third_parties/speecht5/src/Dockerfile.intel_hpu .
```

#### 2.1.2 TTS Service Image

```bash
docker build -t opea/tts:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/tts/src/Dockerfile .
```

### 2.2 Start SpeechT5 and TTS Service

#### 2.2.1 Start SpeechT5 Server

- Xeon

```bash
docker run -p 7055:7055 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/speecht5:latest
```

- Gaudi2 HPU

```bash
docker run -p 7055:7055 --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/speecht5-gaudi:latest
```

#### 2.2.2 Start TTS service

```bash
ip_address=$(hostname -I | awk '{print $1}')

docker run -p 9088:9088 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e TTS_ENDPOINT=http://$ip_address:7055 opea/tts:latest
```

#### 2.2.3 Test

```bash
curl http://localhost:7055/v1/tts -XPOST -d '{"text": "Who are you?"}' -H 'Content-Type: application/json'

# openai protocol compatible
# voice can be 'male' or 'default'
curl http://localhost:9088/v1/audio/speech -XPOST -d '{"input":"Who are you?", "voice": "male"}' -H 'Content-Type: application/json' --output speech.wav
```

## 🚀3. Start Microservice with Docker Compose (Option 3)

Alternatively, you can also start the TTS microservice with Docker Compose.

```bash
export ip_address=$(hostname -I | awk '{print $1}')
# default speecht5 port 7055
export TTS_ENDPOINT=http://$ip_address:7055
# default gptsovits port 9880
# if you want to use gptsovits, please execute the following command instead
# export TTS_ENDPOINT=http://$ip_address:9880
export no_proxy=localhost,$no_proxy

# speecht5 cpu
docker compose -f ../deployment/docker_compose/compose.yaml up speecht5-service tts-speecht5 -d

# speecht5 hpu
docker compose -f ../deployment/docker_compose/compose.yaml up speecht5-gaudi-service tts-speecht5-gaudi -d

# gptsovits cpu
docker compose -f ../deployment/docker_compose/compose.yaml up tts-gptsovits gpt-sovits-service -d

# Test
curl http://localhost:9088/v1/audio/speech -XPOST -d '{"input":"Who are you?"}' -H 'Content-Type: application/json' --output speech.wav
```
