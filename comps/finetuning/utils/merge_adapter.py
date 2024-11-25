# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(base_model_path, adapter_model_path, output_path):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, adapter_model_path)
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False
    merged_model = model.merge_and_unload()
    merged_model.train(False)
    base_model.save_pretrained(output_path, state_dict=merged_model.state_dict())

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
