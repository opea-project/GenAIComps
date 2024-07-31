# Avatar Animation Microservice
The avatar animation model is a combination of two models: Wav2Lip and GAN-based face generator (GFPGAN). The Wav2Lip model is used to generate lip movements from an audio file, and the GFPGAN model is used to generate a high-quality face image from a low-quality face image. The avatar animation microservices takes an audio piece and a low-quality face image/video as input, fuses mel-spectrogram from the audio with frame(s) from the image/video, and generates a high-quality video of the face image with lip movements synchronized with the audio.


# 🚀1. Start Microservice with Docker (option 1)
## 1.1 Build the Docker image
```bash
docker build -t opea/animation:latest -f comps/animation/Dockerfile_hpu .
```

## 1.2. Set environment variables
```bash
export ip_address=$(hostname -I | awk '{print $1}')
export ANIMATION_PORT=7860 
export INFERENCE_MODE='wav2clip+gfpgan'
export CHECKPOINT_PATH='src/Wav2Lip/checkpoints/wav2lip_gan.pth'
export FACE='assets/avatar1.jpg'
export AUDIO='assets/eg3_ref.wav'
export FACESIZE='96'
export OUTFILE='/home/demo/ctao/forks/GenAIComps/comps/animation/outputs/result.mp4'
export GFPGAN_MODEL_VERSION='1.3'
export UPSCALE_FACTOR='1'
export FPS='10.'
```

## 1.3. Run the Docker container
```bash
docker run --privileged --rm -itd --runtime=habana  --cap-add=sys_nice --net=host --ipc=host --name "animation-service" -p 7860:7860 -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker -v $(pwd):$(pwd) -w /home/user/comps/animation -e HABANA_VISIBLE_DEVICES="3" -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e PYTHON=/usr/bin/python3.10 -e INFERENCE_MODE=$INFERENCE_MODE -e CHECKPOINT_PATH=$CHECKPOINT_PATH -e FACE=$FACE -e AUDIO=$AUDIO -e FACESIZE=$FACESIZE -e OUTFILE=$OUTFILE -e GFPGAN_MODEL_VERSION=$GFPGAN_MODEL_VERSION -e UPSCALE_FACTOR=$UPSCALE_FACTOR -e FPS=$FPS opea/animation:latest
```


# 🚀2. Start Microservice with Docker (option 2)
```bash
python /home/user/comps/animation/animation.py --inference_mode $INFERENCE_MODE --checkpoint_path $CHECKPOINT_PATH --face $FACE --audio $AUDIO --outfile $OUTFILE --img_size $FACESIZE -v $GFPGAN_MODEL_VERSION -s $UPSCALE_FACTOR --fps $FPS --only_center_face --bg_upsampler None
```


# 🚀3. Validate Microservice
```bash
python test_animation_server.py
```

