#!/bin/bash

echo "prepare llama-factory for xtune"
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && git fetch origin pull/7519/head:xtune && git checkout xtune && cd ..
rsync -avPr LLaMA-Factory/  .
rm -rf LLaMA-Factory
echo "prepare llama-factory done"