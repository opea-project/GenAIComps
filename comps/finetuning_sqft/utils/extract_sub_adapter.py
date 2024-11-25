import argparse
import os
import re

import torch
from nncf import NNCFConfig
from peft.utils import CONFIG_NAME, WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME

PATTERN = re.compile(r"[[](.*?)[]]", re.S)


def get_width_for_query_prefix(torch_module_to_width, query_module, length=5):
    """
    Get the width for a given query module prefix.

    Args:
        torch_module_to_width (dict): Mapping from torch module to width.
        query_module (str): The query module name.
        length (int, optional): The length of the prefix to match. Default is 5.

    Returns:
        int: The width for the query module prefix.
    """
    query_module_list = query_module.split(".")
    width = next(
        (
            value
            for torch_module, value in torch_module_to_width.items()
            if torch_module.split(".")[:length] == query_module_list[:length]
        ),
        None,
    )
    return width


def main(adapter_model_path, nncf_config, sub_adapter_version, custom_config=None):
    output_dir = os.path.join(adapter_model_path, sub_adapter_version)
    os.makedirs(output_dir, exist_ok=True)
    nncf_config = NNCFConfig.from_json(nncf_config)
    try:
        overwrite_groups = nncf_config["bootstrapNAS"]["training"]["elasticity"]["width"]["overwrite_groups"]
        overwrite_groups_widths = nncf_config["bootstrapNAS"]["training"]["elasticity"]["width"][
            "overwrite_groups_widths"
        ]
        assert len(overwrite_groups) == len(overwrite_groups_widths)
    except Exception:
        raise ValueError("Cannot get the search space in NNCF config.")

    if sub_adapter_version == "maximal":
        subnetwork_config = {idx: space[0] for idx, space in enumerate(overwrite_groups_widths)}
    elif sub_adapter_version == "heuristic":
        subnetwork_config = {idx: space[(len(space) - 1) // 2] for idx, space in enumerate(overwrite_groups_widths)}
    elif sub_adapter_version == "minimal":
        subnetwork_config = {idx: space[-1] for idx, space in enumerate(overwrite_groups_widths)}
    else:
        assert custom_config is not None, "Missing custom subnetwork config."
        assert isinstance(custom_config, list), "Custom config must be a list."
        subnetwork_config = {i: value for i, value in enumerate(custom_config)}

    # Mapping: nncf node -> width
    nncf_node_to_width = {}
    for idx, value in subnetwork_config.items():
        space = overwrite_groups_widths[idx]
        assert min(space) <= value <= max(space)
        cur_dict = {node: value for node in overwrite_groups[idx]}
        nncf_node_to_width.update(cur_dict)

    # Prune adapter model (LoRA low-rank)
    lora_torch_module_to_width = {
        ".".join(re.findall(PATTERN, k)): v for k, v in nncf_node_to_width.items() if "lora_A" in k
    }
    num_module_name_item = list(lora_torch_module_to_width.keys())[0].split(".").index("lora_A")
    # Load adapter weights
    try:
        super_adapter_weights = torch.load(os.path.join(adapter_model_path, WEIGHTS_NAME))
    except:
        from safetensors.torch import load_file
        super_adapter_weights = load_file(os.path.join(adapter_model_path, SAFETENSORS_WEIGHTS_NAME))
    sub_adapter_weights = {}
    for weight_key, weight_tensor in super_adapter_weights.items():
        width = get_width_for_query_prefix(lora_torch_module_to_width, weight_key, length=num_module_name_item)
        if width is not None:
            is_loraA = "lora_A" in weight_key
            new_weight_tensor = weight_tensor[:width].clone() if is_loraA else weight_tensor[:, :width].clone()
        else:
            new_weight_tensor = weight_tensor.clone()
        sub_adapter_weights[weight_key] = new_weight_tensor
    os.makedirs(output_dir, exist_ok=True)
    torch.save(sub_adapter_weights, os.path.join(output_dir, WEIGHTS_NAME))
    config_path = os.path.join(adapter_model_path, CONFIG_NAME)
    os.system(f"cp {config_path} {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge base model and adapter model with additional configurations")
    parser.add_argument('--adapter_model_path', type=str, required=True, help="Path to the adapter model")
    parser.add_argument('--nncf_config', type=str, required=True, help="Path to the NNCF configuration")
    parser.add_argument('--sub_adapter_version', type=str, required=True, help="Sub adapter version")
    parser.add_argument('--custom_config', type=str, default=None, help="Path to custom configuration (optional)")
    args = parser.parse_args()
    main(args.adapter_model_path, args.nncf_config, args.sub_adapter_version, args.custom_config)
