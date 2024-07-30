## 1. Build the Docker image
```bash
docker build -t opea/animation:test -f comps/animation/Dockerfile_hpu .
```

## 2. Set environment variables
```bash
export INFERENCE_MODE='wav2clip_only'
export CHECKPOINT_PATH='src/Wav2Lip/checkpoints/wav2lip_gan.pth'
export FACE='assets/avatar1.jpg'
export AUDIO='assets/eg3_ref.wav'
export FACESIZE='96'
export OUTFILE='outputs/result.mp4'
export GFPGAN_MODEL_VERSION='1.3'
export UPSCALE_FACTOR='1'
export FPS='10.'
```

## 3. Run the Docker container
```bash
docker run --privileged --rm -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker -v $(pwd):$(pwd) -w /home/user/comps/animation -it --runtime=habana -e HABANA_VISIBLE_DEVICES="3" -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e PYTHON=/usr/bin/python3.10 -e INFERENCE_MODE=$INFERENCE_MODE -e CHECKPOINT_PATH=$CHECKPOINT_PATH -e FACE=$FACE -e AUDIO=$AUDIO -e FACESIZE=$FACESIZE -e OUTFILE=$OUTFILE -e GFPGAN_MODEL_VERSION=$GFPGAN_MODEL_VERSION -e UPSCALE_FACTOR=$UPSCALE_FACTOR -e FPS=$FPS --cap-add=sys_nice --net=host --ipc=host opea/animation:test
```

## 4. Run python script
```bash
python /home/user/comps/animation/animation.py --inference_mode $INFERENCE_MODE --checkpoint_path $CHECKPOINT_PATH --face $FACE --audio $AUDIO --outfile $OUTFILE --img_size $FACESIZE -v $GFPGAN_MODEL_VERSION -s $UPSCALE_FACTOR --fps $FPS --only_center_face --bg_upsampler None
```