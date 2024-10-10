# Avatar Animation Microservice

The avatar animation model is a combination of two models: Wav2Lip and GAN-based face generator (GFPGAN). The Wav2Lip model is used to generate lip movements from an audio file, and the GFPGAN model is used to generate a high-quality face image from a low-quality face image. The avatar animation microservices takes an audio piece and a low-quality face image/video as input, fuses mel-spectrogram from the audio with frame(s) from the image/video, and generates a high-quality video of the face image with lip movements synchronized with the audio.

# 🚀1. Start Microservice with Docker (option 1)

## 1.1 Build the Docker image

```bash
git clone https://github.com/opea-project/GenAIComps.git
cd GenAIComps
docker build -t opea/animation:latest -f comps/animation/Dockerfile_hpu .
```

## 1.2. Set environment variables

```bash
export ip_address=$(hostname -I | awk '{print $1}')
export ANIMATION_PORT=7860
export INFERENCE_MODE='wav2clip+gfpgan'
export CHECKPOINT_PATH='/usr/local/lib/python3.10/dist-packages/Wav2Lip/checkpoints/wav2lip_gan.pth'
export FACE='assets/avatar1.jpg'
# export AUDIO='assets/eg3_ref.wav' # audio file path is optional, will use base64str as input if is 'None'
export AUDIO='None'
export FACESIZE=96
export OUTFILE="$(pwd)/comps/animation/outputs/result.mp4"
export GFPGAN_MODEL_VERSION=1.3
export UPSCALE_FACTOR=1
export FPS=10
```

## 1.3. Run the Docker container

<!-- docker run --privileged --rm -itd -->

```bash
docker run --privileged -d --runtime=habana --cap-add=sys_nice --net=host --ipc=host --name "animation-service" -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker -v $(pwd):$(pwd) -w /home/user/comps/animation -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e PYTHON=/usr/bin/python3.10 -e INFERENCE_MODE=$INFERENCE_MODE -e CHECKPOINT_PATH=$CHECKPOINT_PATH -e FACE=$FACE -e AUDIO=$AUDIO -e FACESIZE=$FACESIZE -e OUTFILE=$OUTFILE -e GFPGAN_MODEL_VERSION=$GFPGAN_MODEL_VERSION -e UPSCALE_FACTOR=$UPSCALE_FACTOR -e FPS=$FPS -e ANIMATION_PORT=$ANIMATION_PORT opea/animation:latest
```

# 🚀2. Start Microservice with Python (option 2)

Follow 1.1 and 1.2 steps from the above section to build the Docker image and set the environment variables.

## 2.1. Run the Docker container by overriding the entrypoint

```bash
docker run --privileged --rm -it --runtime=habana --cap-add=sys_nice --net=host --ipc=host --name "animation-service" -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker -v $(pwd):$(pwd) -w /home/user/comps/animation -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e PYTHON=/usr/bin/python3.10 -e INFERENCE_MODE=$INFERENCE_MODE -e CHECKPOINT_PATH=$CHECKPOINT_PATH -e FACE=$FACE -e AUDIO=$AUDIO -e FACESIZE=$FACESIZE -e OUTFILE=$OUTFILE -e GFPGAN_MODEL_VERSION=$GFPGAN_MODEL_VERSION -e UPSCALE_FACTOR=$UPSCALE_FACTOR -e FPS=$FPS -e ANIMATION_PORT=$ANIMATION_PORT --entrypoint /bin/bash opea/animation:latest
```

## 2.2 Inside the container, run the following command to start the microservice

```bash
python3 animation.py --inference_mode $INFERENCE_MODE --checkpoint_path $CHECKPOINT_PATH --face $FACE --audio $AUDIO --outfile $OUTFILE --img_size $FACESIZE -v $GFPGAN_MODEL_VERSION -s $UPSCALE_FACTOR --fps $FPS --only_center_face --bg_upsampler None
```

# 🚀3. Validate Microservice

Once microservice starts, user can use below script to validate the running microservice.

```bash
cd GenAIComps
export ip_address=$(hostname -I | awk '{print $1}')
curl http://${ip_address}:7860/v1/animation -X POST -H "Content-Type: application/json" -d @comps/animation/assets/audio/sample_question.json
```

or

```bash
cd GenAIComps/comps/animation
python3 test_animation_server.py
```

The expected output is a message similar to the following:

```bash
"Status code: 200"
"Check $OUTFILE for the result."
"{'id': '33dd8249228b0e011a33b449af9aa776', 'video_save_path': '.../GenAIComps/comps/animation/outputs/result.mp4'}"
```

Please find "./outputs/result.mp4" as a reference generated video.
