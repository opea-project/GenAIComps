# Avatar Animation Microservice

The avatar animation model is a combination of two models: Wav2Lip and GAN-based face generator (GFPGAN). The Wav2Lip model is used to generate lip movements from an audio file, and the GFPGAN model is used to generate a high-quality face image from a low-quality face image. The avatar animation microservices takes an audio piece and a low-quality face image/video as input, fuses mel-spectrogram from the audio with frame(s) from the image/video, and generates a high-quality video of the face image with lip movements synchronized with the audio.

# 🚀1. Start Microservice with Docker (option 1)

## 1.1 Build the Docker images

### 1.1.1 Wav2Lip Server image

```bash
git clone https://github.com/opea-project/GenAIComps.git
cd GenAIComps
```

- Xeon CPU

```bash
docker build -t opea/wav2lip:latest -f comps/animation/dependency/Dockerfile .
```

- Gaudi2 HPU

```bash
docker build -t opea/wav2lip-gaudi:latest -f comps/animation/dependency/Dockerfile.intel_hpu .
```

### 1.1.2 Animation server image
```bash
docker build -t opea/animation:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/animation/wav2lip/Dockerfile .
```

## 1.2. Set environment variables

```bash
export ip_address=$(hostname -I | awk '{print $1}')
# export DEVICE="cpu"
export DEVICE="hpu"
export WAV2LIP_PORT=7860
export ANIMATION_PORT=9066
export INFERENCE_MODE='wav2lip+gfpgan'
export CHECKPOINT_PATH='/usr/local/lib/python3.10/dist-packages/Wav2Lip/checkpoints/wav2lip_gan.pth'
export FACE='assets/avatar1.jpg'
# export AUDIO='assets/eg3_ref.wav' # audio file path is optional, will use base64str in the post request as input if is 'None'
export AUDIO='None'
export FACESIZE=96
export OUTFILE="$(pwd)/comps/animation/wav2lip/assets/outputs/result.mp4"
export GFPGAN_MODEL_VERSION=1.3
export UPSCALE_FACTOR=1
export FPS=10
```

# 🚀2. Run the Docker container
## 2.1 Run Wav2Lip Microservice
- Xeon CPU

```bash
docker run --privileged -d -p 7860:7860 --ipc=host --name "wav2lip-service" -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker -v $(pwd):$(pwd) -w /home/user/comps/animation -e PYTHON=/usr/bin/python3.11 -e DEVICE=$DEVICE -e INFERENCE_MODE=$INFERENCE_MODE -e CHECKPOINT_PATH=$CHECKPOINT_PATH -e FACE=$FACE -e AUDIO=$AUDIO -e FACESIZE=$FACESIZE -e OUTFILE=$OUTFILE -e GFPGAN_MODEL_VERSION=$GFPGAN_MODEL_VERSION -e UPSCALE_FACTOR=$UPSCALE_FACTOR -e FPS=$FPS -e WAV2LIP_PORT=$WAV2LIP_PORT opea/wav2lip:latest
```

- Gaudi2 HPU

```bash
docker run --privileged -d --runtime=habana --cap-add=sys_nice --net=host --ipc=host --name "wav2lip-gaudi-service" -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker -v $(pwd):$(pwd) -w /home/user/comps/animation -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e PYTHON=/usr/bin/python3.10 -e DEVICE=$DEVICE -e INFERENCE_MODE=$INFERENCE_MODE -e CHECKPOINT_PATH=$CHECKPOINT_PATH -e FACE=$FACE -e AUDIO=$AUDIO -e FACESIZE=$FACESIZE -e OUTFILE=$OUTFILE -e GFPGAN_MODEL_VERSION=$GFPGAN_MODEL_VERSION -e UPSCALE_FACTOR=$UPSCALE_FACTOR -e FPS=$FPS -e WAV2LIP_PORT=$WAV2LIP_PORT opea/wav2lip-gaudi:latest
```

## 2.2 Run Animation Microservice
```bash
docker run -d -p 9066:9066 --ipc=host --name "animation-service" -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e WAV2LIP_ENDPOINT=http://$ip_address:7860 opea/animation:latest
```

# 🚀3. Validate Microservice

Once microservice starts, user can use below script to validate the running microservice.

## 3.1 Validate Wav2Lip service

```bash
cd GenAIComps/comps/animation/wav2lip
python3 dependency/check_wav2lip_server.py
```

## 3.2 Validate Animation service

```bash
cd GenAIComps
export ip_address=$(hostname -I | awk '{print $1}')
curl http://${ip_address}:7860/v1/animation -X POST -H "Content-Type: application/json" -d @comps/animation/wav2lip/assets/audio/sample_question.json
```

or

```bash
cd GenAIComps/comps/animation/wav2lip
python3 check_animation_server.py
```

The expected output is a message similar to the following:

```bash
"Status code: 200"
"Check $OUTFILE for the result."
"{'id': '33dd8249228b0e011a33b449af9aa776', 'video_save_path': '.../GenAIComps/comps/animation/wav2lip/assets/outputs/result.mp4'}"
```

Please find "./outputs/result.mp4" as a reference generated video.
