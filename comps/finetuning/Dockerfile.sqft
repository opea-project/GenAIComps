# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Use the same python version with ray
FROM python:3.10.14

ARG HF_TOKEN

ENV HF_TOKEN=$HF_TOKEN

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

COPY comps /home/user/comps

RUN chown -R user /home/user/comps/finetuning

USER user

ENV PATH=$PATH:/home/user/.local/bin

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --no-cache-dir intel-extension-for-pytorch && \
    python -m pip install --no-cache-dir oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/ && \
    python -m pip install --no-cache-dir -r /home/user/comps/finetuning/requirements.txt

WORKDIR /home/user/comps/finetuning

RUN git clone https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning.git && \
    cp -r Hardware-Aware-Automated-Machine-Learning/SQFT/patches /home/user/comps/finetuning/patches && \
    rm -rf Hardware-Aware-Automated-Machine-Learning && \
    mkdir third_party

# Clone and set up transformers
RUN git clone https://github.com/huggingface/transformers.git third_party/transformers && \
    cd third_party/transformers && \
    git checkout v4.44.2 && \
    git apply --ignore-space-change --ignore-whitespace /home/user/comps/finetuning/patches/transformers-v4.44.2.patch && \
    pip install -e .

# Clone and set up peft
RUN git clone https://github.com/huggingface/peft.git third_party/peft && \
    cd third_party/peft && \
    git checkout v0.10.0 && \
    git apply --ignore-space-change --ignore-whitespace /home/user/comps/finetuning/patches/peft-v0.10.0.patch && \
    pip install -e .

# Clone and set up nncf
RUN git clone https://github.com/openvinotoolkit/nncf.git third_party/nncf && \
    cd third_party/nncf && \
    git checkout f143e1c && \
    git apply --ignore-space-change --ignore-whitespace /home/user/comps/finetuning/patches/nncf-f143e1c.patch && \
    pip install -e .

ENV PYTHONPATH=$PYTHONPATH:/home/user

RUN echo PKGPATH=$(python3 -c "import pkg_resources; print(pkg_resources.get_distribution('oneccl-bind-pt').location)") >> run.sh && \
    echo 'export LD_LIBRARY_PATH=$PKGPATH/oneccl_bindings_for_pytorch/opt/mpi/lib/:$LD_LIBRARY_PATH' >> run.sh && \
    echo 'source $PKGPATH/oneccl_bindings_for_pytorch/env/setvars.sh' >> run.sh && \
    echo ray start --head --dashboard-host=0.0.0.0 >> run.sh && \
    echo export RAY_ADDRESS=http://localhost:8265 >> run.sh && \
    echo python finetuning_service.py >> run.sh

CMD bash run.sh
