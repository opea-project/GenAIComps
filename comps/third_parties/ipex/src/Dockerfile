# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Most of this Dockerfile originates from https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/llm/Dockerfile

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} AS base

RUN apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --fix-missing \
    ca-certificates \
    curl \
    g++-12 \
    gcc-12 \
    git \
    make \
    numactl \
    wget

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100

WORKDIR /root

RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda-forge/miniforge/releases/download/24.7.1-2/Miniforge3-24.7.1-2-Linux-x86_64.sh && \
    bash miniforge.sh -b -p ./miniforge3 && \
    rm miniforge.sh

# --build-arg COMPILE=ON to compile from source
FROM base AS dev
ARG COMPILE
RUN git clone --depth 1 https://github.com/intel/intel-extension-for-pytorch.git
RUN . ~/miniforge3/bin/activate && conda create -y -n compile_py310 python=3.10 && conda activate compile_py310 && \
    cd intel-extension-for-pytorch/examples/cpu/llm && \
    export CC=gcc && export CXX=g++ && \
    if [ -z ${COMPILE} ]; then bash tools/env_setup.sh 14; else bash tools/env_setup.sh 10; fi && \
    unset CC && unset CXX

FROM base AS deploy
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --fix-missing \
    google-perftools \
    net-tools \
    openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    if [ -f /etc/apt/apt.conf.d/proxy.conf ]; then rm /etc/apt/apt.conf.d/proxy.conf; fi

COPY --from=dev /root/intel-extension-for-pytorch/examples/cpu/llm ./llm
COPY --from=dev /root/intel-extension-for-pytorch/tools/get_libstdcpp_lib.sh ./llm/tools
RUN . ~/miniforge3/bin/activate && conda create -y -n py310 python=3.10 && conda activate py310 && \
    cd /usr/lib/x86_64-linux-gnu/ && ln -s libtcmalloc.so.4 libtcmalloc.so && cd && \
    cd ./llm && \
    bash tools/env_setup.sh 9 && \
    python -m pip cache purge && \
    mv ./oneCCL_release /opt/oneCCL && \
    chown -R root:root /opt/oneCCL && \
    sed -i "s|ONECCL_PATH=.*|ONECCL_PATH=/opt/oneCCL|" ./tools/env_activate.sh && \
    pip install --no-cache-dir backoff fastapi uvicorn
ARG PORT_SSH=22
RUN mkdir /var/run/sshd && \
    sed -i "s/#Port.*/Port ${PORT_SSH}/" /etc/ssh/sshd_config && \
    echo "service ssh start" >> /root/.bashrc && \
    ssh-keygen -b 4096 -f /root/.ssh/id_rsa -N "" && \
    mv /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
    echo "Host *\n    Port ${PORT_SSH}\n    IdentityFile /root/.ssh/id_rsa\n    StrictHostKeyChecking no" > /root/.ssh/config
EXPOSE ${PORT_SSH}
COPY ./comps/third_parties/ipex/src/ipex_inference.py /root
COPY ./comps/third_parties/ipex/src/openai_protocol.py /root
COPY ./comps/third_parties/ipex/src/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
