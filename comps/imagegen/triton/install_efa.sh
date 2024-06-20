#!/bin/bash -ex

DEFAULT_EFA_INSTALLER_VER=1.29.0
efa_installer_version=${1:-$DEFAULT_EFA_INSTALLER_VER}

tmp_dir=$(mktemp -d)
wget -nv https://efa-installer.amazonaws.com/aws-efa-installer-$efa_installer_version.tar.gz -P $tmp_dir
tar -xf $tmp_dir/aws-efa-installer-$efa_installer_version.tar.gz -C $tmp_dir
pushd $tmp_dir/aws-efa-installer
case $(. /etc/os-release ; echo -n $ID) in
    rhel)
        # we cannot install dkms packages on RHEL images due to OCP rules
        rm -f RPMS/RHEL8/x86_64/dkms*.rpm
    ;;
    tencentos)
        patch -f -p1 -i /tmp/tencentos_efa_patch.txt --reject-file=tencentos_efa_patch.rej --no-backup-if-mismatch
    ;;
esac
./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify
popd
rm -rf $tmp_dir
