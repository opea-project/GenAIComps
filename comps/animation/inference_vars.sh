# Environment variables
export PT_HPU_LAZY_MODE=0
export PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1

# Wav2Lip, GFPGAN
python3 Wav2Lip-GFPGAN/inference_wav2lip+gfpgan.py \
--inference_mode $INFERENCE_MODE \
--checkpoint_path $CHECKPOINT_PATH \
--face $FACE \
--audio $AUDIO \
--outfile $OUTFILE \
--img_size $((FACESIZE)) \
-v $GFPGAN_MODEL_VERSION \
-s $((UPSCALE_FACTOR)) \
--fps $FPS \
--only_center_face \
--bg_upsampler None

# --face /home/ubuntu/ctao/Wav2Lip-GFPGAN/inputs/7970355-hd_1080_1920_25fps.mp4 \
# -s --upscale can be 1 instead of 2
