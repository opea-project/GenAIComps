import os
import json

try:
    from nncf import NNCFConfig
    from nncf.experimental.torch import sqft
    is_nncf_available = True
except ImportError:
    is_nncf_available = False


NNCF_CONFIG_TEMPLATE = {
    "input_info": [
        {
            "sample_size": [1, 256],
            "type": "long",
            "keyword": "input_ids"
        },
        {
            "sample_size": [1, 256],
            "type": "long",
            "keyword": "attention_mask"
        }
    ],
    "SQFT": {
        "training": {
            "algorithm": "nls",
            "elasticity": {
                "available_elasticity_dims": ["width"],
                "width": {
                    "overwrite_groups": [],
                    "overwrite_groups_widths": []
                }
            }
        }
    }
}


def add_lr_epochs(nncf_config, learning_rate=3e-4, num_train_epochs=3):
    """Add learning rate and epochs to the NNCF configuration.

    Args:
        nncf_config (dict): The NNCF configuration dictionary.
        learning_rate (float): The initial learning rate to set.
        num_epochs (int): The number of epochs to set.

    Returns:
        dict: The updated NNCF configuration.
    """
    overwrite_groups_widths = nncf_config["SQFT"]["training"]["elasticity"]["width"]["overwrite_groups_widths"]
    # Add learning rate and epochs to the configuration
    nncf_config["SQFT"]["training"]["schedule"] = {
        "list_stage_descriptions": [
            {
                "train_dims": ["width"],
                "width_indicator": max([len(widths) for widths in overwrite_groups_widths]),
                "init_lr": learning_rate,
                "epochs": num_train_epochs,
                "epochs_lr": num_train_epochs,
            }
        ]
    }
    return nncf_config


def get_model_paths(model, target_module_name):
    """
    Find all paths to the target layer in the model.

    Args:
        model (torch.nn.Module): The model to search.
        target_module_name (str): The name of the target layer.

    Returns:
        list: A list of paths to the target layer.
    """
    def find_layers(module, target_module_name, path, paths):
        for name, sub_module in module.named_children():
            new_path = f"{path}/{sub_module.__class__.__name__}[{name}]"
            if target_module_name in name:
                # Check if 'lora_A' is in the sub_module's children
                for sub_name, _ in sub_module.named_children():
                    if "lora_A" in sub_name:
                        paths.append(f"{new_path}/ModuleDict[lora_A]/NNCFLinear[default]/linear_0")
            find_layers(sub_module, target_module_name, new_path, paths)

    base_path = model.__class__.__name__
    paths = []
    find_layers(model, target_module_name, base_path, paths)
    return paths


def create_sqft_nncf_config(
    config,
    model,
    target_module_groups=None,
    search_space=None
):
    """Load and preprocess the NNCF configuration file.

    Returns:
        NNCFConfig: The preprocessed NNCF configuration object.
    """
    if not is_nncf_available:
        raise NotImplementedError("NNCF is not installed. Please install it for enabling NLS algorithm.")
    if search_space is None and target_module_groups:
        raise ValueError("Neural LoRA search is enabled, `search_space` and `target_module_groups` must be provided.")
    # The NNCF Config will be automatically generated based on `target_module_groups` and `search_space`.
    num_hidden_layers = model.config.num_hidden_layers
    nncf_config_dict = NNCF_CONFIG_TEMPLATE
    overwrite_groups = []
    for group in target_module_groups:
        group_paths = []
        for module in group:
            target_layer_name = module
            paths = get_model_paths(model, target_layer_name)
            assert paths, f"No paths found for module {module}"
            group_paths.append(paths)
        # Transpose the list of lists to combine paths by their positions
        transposed_paths = list(zip(*group_paths))
        overwrite_groups.extend([list(path_group) for path_group in transposed_paths])
    nncf_config_dict["SQFT"]["training"]["elasticity"]["width"]["overwrite_groups"] = overwrite_groups

    overwrite_groups_widths = []
    for space in search_space:
        space = [int(width) for width in space.split(",")]
        overwrite_groups_widths.extend([space] * num_hidden_layers)
    nncf_config_dict["SQFT"]["training"]["elasticity"]["width"]["overwrite_groups_widths"] = overwrite_groups_widths
    assert len(overwrite_groups) == len(overwrite_groups_widths)
    nncf_config_dict = add_lr_epochs(
        nncf_config_dict,
        learning_rate=config["Training"]["learning_rate"],
        num_train_epochs=config["Training"]["epochs"]
    )
    nncf_config = NNCFConfig.from_dict(nncf_config_dict)

    nncf_config["log_dir"] = config["General"]["output_dir"]
    os.makedirs(nncf_config["log_dir"], exist_ok=True)
    with open(os.path.join(nncf_config["log_dir"], "nncf_config.json"), "w") as f:
        json.dump(nncf_config, f, indent=4)
    return nncf_config
