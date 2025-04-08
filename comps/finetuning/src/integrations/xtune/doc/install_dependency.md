# Install linux kernel

You can refer to https://dgpu-docs.intel.com/driver/client/overview.html .
Below is example of install driver on ubuntu 22.04 for Arc 770.

```bash
# Install kernel
apt install linux-image-6.5.0-35-generic
apt install linux-headers-6.5.0-35-generic
apt install linux-modules-6.5.0-35-generic
apt install linux-modules-extra-6.5.0-35-generic
```

```bash
# Update boot to linux-image-6.5.0-35-generic
update grub
reboot
# check your kernel version
uname -r
```

```bash
$ vim /etc/modprobe.d/blacklist.conf
# add “blacklist ast” at the bottom
$ update-initramfs -u
```

# Install Driver

```bash
# Install the Intel graphics GPG public key
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

# Configure the repositories.intel.com package repository
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

# Update the package repository metadata
apt update

# Install i915
apt install intel-i915-dkms
reboot
```

```bash
# Install the compute-related packages
apt-get install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo intel-gsc
apt-get install -y libze-dev intel-ocloc
apt-get install -y intel-level-zero-gpu-raytracing xpu-smi
```

To verify that the kernel and compute drivers are installed and functional, run clinfo:

```bash
clinfo | grep "Device Name"
Device Name                                     Intel(R) Arc(TM) A770 Graphics
```

You should see the Intel graphics product device names listed. If they do not appear, ensure you have permissions to access `/dev/dri/rendered*`. This typically requires your user to be in the render group:

```bash
sudo gpasswd -a ${USER} render
newgrp render
```

## Optional:

Configure the window system to wayland

```bash
$ vim /etc/gdm3/custom.conf
# Set WaylandEnable=true
$ update-initramfs
```
