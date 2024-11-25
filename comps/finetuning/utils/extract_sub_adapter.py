import os
import re

import torch

from peft.utils import CONFIG_NAME, WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME

try:
    from nncf import NNCFConfig
    is_nncf_available = True
except ImportError:
    is_nncf_available = False


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


def main(adapter_model_path, nncf_config, adapter_version, custom_config=None):
    if not is_nncf_available:
        raise NotImplementedError("NNCF is not installed. Please install it.")
    output_dir = os.path.join(adapter_model_path, adapter_version)
    os.makedirs(output_dir, exist_ok=True)
    nncf_config = NNCFConfig.from_json(nncf_config)
    try:
        groups = nncf_config["SQFT"]["training"]["elasticity"]["width"]["overwrite_groups"]
        groups_widths = nncf_config["SQFT"]["training"]["elasticity"]["width"]["overwrite_groups_widths"]
        assert len(groups) == len(groups_widths)
    except Exception:
        raise ValueError("Cannot get the search space in NNCF config.")

    if adapter_version == "maximal":
        subnetwork_config = {idx: space[0] for idx, space in enumerate(groups_widths)}
    elif adapter_version == "heuristic":
        subnetwork_config = {idx: space[(len(space) - 1) // 2] for idx, space in enumerate(groups_widths)}
    elif adapter_version == "minimal":
        subnetwork_config = {idx: space[-1] for idx, space in enumerate(groups_widths)}
    else:
        assert custom_config is not None, "Missing custom subnetwork config."
        assert isinstance(custom_config, list), "Custom config must be a list."
        subnetwork_config = {i: value for i, value in enumerate(custom_config)}

    # Mapping: nncf node -> width
    nncf_node_to_width = {}
    for idx, value in subnetwork_config.items():
        space = groups_widths[idx]
        assert min(space) <= value <= max(space)
        cur_dict = {node: value for node in groups[idx]}
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
