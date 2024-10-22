# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import os

import yaml
from typing import Dict, List, Any


def replace_env_vars(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: replace_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_env_vars(v) for v in data]
    elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
        env_var = data[2:-1]
        return os.getenv(env_var, "")
    else:
        return data

def convert_args_to_command(args: List[str]) -> str:
    command_parts = []
    for arg in args:
        if isinstance(arg, dict):
            for k, v in arg.items():
                command_parts.append(f"{k} {v}")
        elif isinstance(arg, str):
            command_parts.append(arg.replace(':', ' '))
    return " ".join(command_parts)

def convert_resources(resources: Dict[str, Any]) -> Dict[str, Any]:
    converted_resources = {}
    for key, value in resources.items():
        if key == 'hpu':
            # converted_resources['hpus'] = value
            pass
        elif key == 'cpu':
            converted_resources['cpus'] = value
        elif key == 'memory':
            converted_resources['memory'] = value
    return converted_resources

def extract_options(options: List[Any]) -> Dict[str, Any]:
    extracted_options = {}
    for option in options:
        if isinstance(option, dict):
            for k, v in option.items():
                if k == 'cap_add':
                    extracted_options[k] = [v] if isinstance(v, str) else v
                else:
                    extracted_options[k] = v
    return extracted_options

def build_docker_compose(input_data: Dict) -> Dict:
    docker_compose = {
        'version': '3.8',
        'services': {}
    }

    global_envs = input_data.get('global_envs',{})

    for service in input_data.get('micro_services', []) + input_data.get('mega_service', []):
        service_name = service['service_name']
        service_config = {
            'image': service['image'],
            'ports': service.get('ports', []),
            'volumes': service.get('volumes', []),
            'environment': global_envs.copy()
        }

        for env in service.get('envs', []):
            if isinstance(env, list) and len(env) == 2:
                service_config['environment'][env[0]] = env[1]
            elif isinstance(env, dict):
                service_config['environment'].update(env)

        if 'dependencies' in service:
            service_config['depends_on'] = service['dependencies']

        if 'replicas' in service:
            service_config['deploy'] = {'replicas': service['replicas']}
        if 'resources' in service:
            service_config['deploy']['resources'] = {'limits': convert_resources(service.get('resources', {}))}
        if 'options' in service:
            for option in service['options']:
                for key,value in option.items():
                    if key == 'cap_add':
                        service_config[key] = [value] if isinstance(value, str) else value
                    else:
                        service_config[key] = value

        if 'args' in service:
            service_config['command'] = convert_args_to_command(service['args'])

        docker_compose['services'][service_name] = service_config

    return docker_compose


def convert_to_docker_compose(input_yaml_path: str, output_file: str):
    with open(input_yaml_path, 'r') as file:
        input_data = yaml.safe_load(file)

    input_data = replace_env_vars(input_data)

    docker_compose_data = build_docker_compose(input_data)

    with open(output_file, 'w') as file:
        yaml.dump(docker_compose_data, file, default_flow_style=False)

    print("Docker Compose file generated:", output_file)



