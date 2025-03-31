## Requirements

We utilize the code base of [CoOp](https://github.com/KaiyangZhou/CoOp). Please follow their instructions to prepare the environment and datasets.

```python
conda create -y -n clip_adapter python=3.10
conda activate clip_adapter
cd Dassl
# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install dependencies
pip install -r requirements.txt

# Install dassl library (no need to re-build if the source code is modified)
python setup.py develop

cd ..

# Install dependencies for clip
pip install -r requirements.txt
pip install transformers
export HF_ENDPOINT=https://hf-mirror.com
```

# Prepare Dataset

see [doc](../doc/Prepare_dataset.md)

## Get Started

```python
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
# run clip_adapter
# run with huggingface transformers backbone
# run clip_adapter
bash scripts/CLIP_finetune/clip_adapter_hf.sh caltech101 vit_b16 0
# run clip_adapter and do val/train acc cal every 10 epoch
bash scripts/CLIP_finetune/clip_adapter_hf.sh caltech101 vit_b16 10
# run clip_full_finetune
bash scripts/CLIP_finetune/clip_fullfinetune_hf.sh caltech101 vit_b16 0
# run clip_bias
bash scripts/CLIP_finetune/clip_bias_hf.sh caltech101 vit_b16 0
# run clip_prompt with 1 prompt length and use Deep VPT
bash scripts/CLIP_finetune/clip_prompt_hf.sh caltech101 vit_b16 1 True 0
# run clip_prompt with 2 prompt length and don't use Deep VPT
bash scripts/CLIP_finetune/clip_prompt_hf.sh caltech101 vit_b16 2 False 0
# run clip_prompt with 1 prompt length and use Deep VPT using mini-imagenet dataset
bash scripts/CLIP_finetune/clip_prompt_hf.sh mini_imagenet vit_b16 1 True 0
# run clip_adapter using flickr30k dataset
bash scripts/CLIP_finetune/clip_adapter_hf.sh flickr30k vit_b16 0

# checkpoint will save to output/$METHOD/$MODEL/$DATASET
# you can set `export CLIP_DEBUG=1` to remove checkpoint
```

# config yaml for clip_bias

```python
we use yaml to config param.
e.g.       ./configs/clip_finetune/vit_b16_bias_example.yaml
BIAS_TERMS:             which layer's bias you want to tune, default to tune all bias layer, in this config we tune the layer with attn or mlp in name
BIAS_TERMS_EXCLUDE      which layer's bias you don't need to tune, in this config we don't tune text_encoder
```

## code structure

```python
./scripts/CLIP_finetune contains the scripts we use to run
./trainers contains model related code
```

## run on A770

```python
For oneapi & pytorch whl & driver info, please check with the internal AI team realse WIKI(https://wiki.ith.intel.com/pages/viewpage.action?pageId=1786921649)
# download oneapi
    # download and install public oneapi
    Plearse refer to https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
    # download and install internal oneapi
    wget https://af01p-sc.devtools.intel.com/artifactory/satgoneapi-or-local/products/BaseKit/2024.2.1/packages/l_BaseKit_p_2024.2.1.100/webimage/l_BaseKit_p_2024.2.1.100_offline.sh
    wget https://af-satgoneapi.devtools.intel.com/artifactory/satgoneapi-or-local/products/intel-pti-dev/0.9.0/packages/l_intel-pti-dev_p_0.9.0.38/webimage/l_intel-pti-dev_p_0.9.0.38_offline.sh
    sudo bash l_BaseKit_p_2024.2.1.100_offline.sh -a --ignore-errors --silent --eula accept
    sudo bash l_intel-pti-dev_p_0.9.0.38_offline.sh -a --ignore-errors --silent --eula accept

# install intel CA
# Note: for internal client only
sudo -s
wget --no-proxy --no-check-certificate http://certificates.intel.com/repository/certificates/Intel%20Root%20Certificate%20Chain%20Base64.zip && unzip -o ./"Intel Root Certificate Chain Base64.zip" -d /usr/local/share/ca-certificates
wget --no-check-certificate http://certificates.intel.com/repository/certificates/IntelSHA2RootChain-Base64.zip && unzip -o ./"IntelSHA2RootChain-Base64.zip" -d /usr/local/share/ca-certificates
wget --no-check-certificate http://certificates.intel.com/repository/certificates/TrustBundles/IntelSHA384TrustChain-Base64.zip && unzip -o ./"IntelSHA384TrustChain-Base64.zip" -d /usr/local/share/ca-certificates
update-ca-certificates

# install driver
    # install public driver
    Please refer to https://dgpu-docs.intel.com/driver/installation-rolling.html

    # install internal driver
    # Note: for internal client only
    curl --noproxy '*' -H "Authorization: Basic d3Vrbjp3dWtuMTIz" -so ~/install_umd.sh https://osgc-sh-moon.sh.intel.com/api/v1/drvinfo/umd/?v=hotfix_agama-ci-devel-950.16
    curl --noproxy '*' -H "Authorization: Basic d3Vrbjp3dWtuMTIz" -so ~/install_kmd.sh https://osgc-sh-moon.sh.intel.com/api/v1/drvinfo/kmd/?v=hotfix_agama-ci-devel-950.16
    chmod 777 *.sh
    # install driver  --- install kmd
    bash ~/install_kmd.sh
    sudo DEBIAN_FRONTEND=noninteractive apt-get -y --allow-downgrades --reinstall install ~/hotfix_agama-ci-devel-950.16/*.deb
    rm -rf ~/hotfix_agama-ci-devel-950.16
    sudo reboot

    # install driver  --- install umd
    bash ~/install_umd.sh
    sudo dpkg -i --force-all ~/hotfix_agama-ci-devel-950.16/*.deb
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -f -y && sudo apt autoremove -y
    sudo usermod -a -G render <your user>
    su <your user>
    # check if you can see the GPU
    clinfo -l
        gta@DUT6269DG2FRD:~$ clinfo -l
        Platform #0: Intel(R) OpenCL Graphics
        `-- Device #0: Intel(R) Arc(TM) A770 Graphics

# download IPEX relate whl
mkdir whl
cd whl
wget  -r -np -c -nH -nd https://ubit-artifactory-sh.intel.com/artifactory/aipc_releases-sh-local/gpu-new/releases/2024.2/IPEX_2.5.10+xpu/preview/py310/
pip install --force-reinstall *.whl
source /opt/intel/oneapi/setvars.sh
source /opt/intel/oneapi/pti/latest/env/vars.sh

# install transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.38.1
wget https://gfx-assets.sh.intel.com/artifactory/list/gfx-osgc-assets-sh/aitest/pytorch/profile_patch/4.38.1/artifacts/profile_patch
git apply profile_patch
python setup.py install

# run with A770
# run with huggingface transformers backbone
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
bash scripts/CLIP_finetune/clip_adapter_hf.sh caltech101 vit_b16 0 XPU
bash scripts/CLIP_finetune/clip_fullfinetune_hf.sh caltech101 vit_b16 0 XPU
bash scripts/CLIP_finetune/clip_bias_hf.sh caltech101 vit_b16 0 XPU
bash scripts/CLIP_finetune/clip_prompt_hf.sh caltech101 vit_b16 1 True 0 XPU

```

## run on A770 with DDP

```python
source /opt/intel/oneapi/setvars.sh
source /opt/intel/oneapi/pti/latest/env/vars.sh
# install HVD
git clone https://github.com/intel-innersource/frameworks.ai.horovod.git
cd frameworks.ai.horovod/
git checkout r0.28.1.6
git checkout 0364ddfcf4eb978b97f493c7c36e9b45223c0a81
git submodule init && git submodule update
CXX=icpx     CC=icx     HOROVOD_GPU=DPCPP     HOROVOD_WITHOUT_MXNET=1     HOROVOD_WITHOUT_PYTORCH=0     HOROVOD_WITHOUT_TENSORFLOW=1     HOROVOD_WITHOUT_GLOO=1     HOROVOD_GPU_OPERATIONS=CCL     HOROVOD_CPU_OPERATIONS=CCL     HOROVOD_WITH_MPI=1 python setup.py bdist_wheel
cd dist/
pip install *.whl


export NEOReadDebugKeys=1
export DisableScratchPages=0
export CCL_ATL_TRANSPORT=ofi

bash scripts/CLIP_finetune/clip_adapter_hf_mutiXPU.sh caltech101 vit_b16 0 XPU
```

# use optuna to automatic get the best param

You can use optuna(https://github.com/optuna/optuna) to automatic tune the hyperparameter.
We only support turn bs and lr.
You can set the bs and lr in yaml, such as ./configs/clip_finetune/vit_b16_opt.yaml

```python
# turn on optuna in A100
bash scripts/clip_finetune/clip_adapter_hf.sh caltech101 vit_b16 0 cuda 1
# turn on optuna in A770
bash scripts/clip_finetune/clip_adapter_hf.sh caltech101 vit_b16 0 XPU 1
```

# problem

```bash
if you hit below problem with DDP,
    [1] [1728626096.870265672] DUT7113ATSM:rank1.python: Reading from remote process' memory failed. Disabling CMA support
    [1] DUT7113ATSM:rank1: Assertion failure at psm3/ptl_am/ptl.c:210: nbytes == req->req_data.recv_msglen
You can run
    echo 0 >> /proc/sys/kernel/yama/ptrace_scope
This issue is caused by PYTORCHDGQ-4236(https://jira.devtools.intel.com/browse/PYTORCHDGQ-4236)
```
