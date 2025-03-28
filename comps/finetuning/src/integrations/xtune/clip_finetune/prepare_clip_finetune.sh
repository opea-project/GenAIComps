#!/bin/bash

echo "prepare dassl for xtune"
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git dassl
cd dassl && git fetch origin pull/72/head:xtune && git checkout xtune && cd .. && mv dassl clip_finetune/
echo "dassl done"