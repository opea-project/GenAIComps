# Use a base image
FROM python:3.11-slim

# Set environment variables
ENV LANG=en_US.UTF-8
ENV PYTHONPATH=/home:/home/user
ENV PYTHON=/usr/bin/python3.11

# Install dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    yasm \
    libx264-dev

# TO-DO: Add a user for security

# Install GenAIComps
RUN mkdir -p /home/user/comps
COPY comps /home/user/comps
COPY comps/animation/entrypoint.sh /usr/local/bin/entrypoint.sh

# Install ppmpeg with x264 software codec
RUN git clone https://github.com/FFmpeg/FFmpeg.git /home/user/comps/animation/FFmpeg
WORKDIR /home/user/comps/animation/FFmpeg
RUN ./configure --enable-gpl --enable-libx264 && \
    make -j$(nproc-1) && \
    make install && \
    hash -r
RUN chmod +x $(which ffmpeg)

# # Install Wav2Lip-GFPGAN from github
# RUN git clone https://github.com/ajay-sainy/Wav2Lip-GFPGAN.git /home/user/comps/animation/src
# # Rename subfolders "Wav2Lip" and "GFPGAN"
# RUN mv /home/user/comps/animation/src/Wav2Lip-master /home/user/comps/animation/src/Wav2Lip && \
#     mv /home/user/comps/animation/src/GFPGAN-master /home/user/comps/animation/src/GFPGAN

# # Set the PYTHONPATH environment variable
# RUN touch /home/user/comps/animation/src/Wav2Lip/__init__.py
# RUN touch /home/user/comps/animation/src/GFPGAN/__init__.py
# ENV PYTHONPATH="/home/user/comps/animation/src:/home/user/comps/animation/src/Wav2Lip:/home/user/comps/animation/src/GFPGAN:$PYTHONPATH"

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Wav2Lip from pip
RUN pip install --no-deps Wav2Lipy
RUN rm /usr/local/lib/python3.10/dist-packages/Wav2Lip/__init__.py && touch /usr/local/lib/python3.10/dist-packages/Wav2Lip/__init__.py
ENV PYTHONPATH="/usr/local/lib/python3.10/dist-packages/Wav2Lip:$PYTHONPATH"

# Install GFPGAN from pip
RUN pip install --no-deps gfpgan
RUN touch /usr/local/lib/python3.10/dist-packages/gfpgan/__init__.py
ENV PYTHONPATH="/usr/local/lib/python3.10/dist-packages/gfpgan:$PYTHONPATH"

# Download pre-trained models
# Install gdown
RUN pip install gdown
WORKDIR /usr/local/lib/python3.10/dist-packages
# RUN wget https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth -O Wav2Lip/face_detection/detection/sfd/s3fd.pth
RUN gdown https://drive.google.com/uc?id=1wwts8U4Wx2lpLOI9uGcUB-TLmKRIFnP8 -O Wav2Lip/face_detection/detection/sfd/s3fd.pth
RUN mkdir -p Wav2Lip/checkpoints
RUN gdown https://drive.google.com/uc?id=1mIKfu_onFKbkbrq6cKVRBblXe5bNMhd9 -O Wav2Lip/checkpoints/wav2lip.pth
RUN gdown https://drive.google.com/uc?id=1DD7dtUfNWqNoW-2Gnp78B4PyaiJ5ybhP -O Wav2Lip/checkpoints/wav2lip_gan.pth
RUN wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P gfpgan/experiments/pretrained_models

# Install pip dependencies
RUN pip install -r /home/user/comps/animation/requirements.txt

# Custom patches
# Modify the degradations.py file to import rgb_to_grayscale from torchvision.transforms.functional
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py

# Modify the core.py file to include 'hpu' in the device check
RUN sed -i "s/if 'cpu' not in device and 'cuda' not in device:/if 'cpu' not in device and 'cuda' not in device and 'hpu' not in device:/" /usr/local/lib/python3.10/dist-packages/Wav2Lip/face_detection/detection/core.py

# To be compatible with librosa==0.10.2, instead of librosa==0.7.0 because the co-dependency numba==0.48 cannot be pip installed
RUN sed -i 's/hp.sample_rate, hp.n_fft/sr=hp.sample_rate, n_fft=hp.n_fft/' /usr/local/lib/python3.10/dist-packages/Wav2Lip/audio.py

# Set the working directory
WORKDIR /home/user/comps/animation

# Define the command to run when the container starts
RUN chmod +x /usr/local/bin/entrypoint.sh
ENV DEVICE="cpu"
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]