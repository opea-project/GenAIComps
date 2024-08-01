# /bin/bash
# Download models
MODEL_REPO=https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned
echo "Please wait for model download..."
git lfs install &&  git clone ${MODEL_REPO} /home/user/model/Video-LLaMA-2-7B-Finetuned
# rm Video-LLaMA-2-7B-Finetuned/AL*.pth Video-LLaMA-2-7B-Finetuned/imagebind_huge.pth
python server.py