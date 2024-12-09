# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class OpeaComponent:
    """
    The OpeaComponent class serves as the base class for all components in the GenAIComps.
    It provides a unified interface and foundational attributes that every derived component inherits and extends.

    Attributes:
        name (str): The name of the component.
        type (str): The type of the component (e.g., 'retriever', 'embedding', 'reranking', 'llm', etc.).
        description (str): A brief description of the component's functionality.
        config (dict): A dictionary containing configuration parameters for the component.
    """

    def __init__(self, name: str, type: str, description: str, config: dict = None):
        """
        Initializes an OpeaComponent instance with the provided attributes.

        Args:
            name (str): The name of the component.
            type (str): The type of the component.
            description (str): A brief description of the component.
            config (dict, optional): Configuration parameters for the component. Defaults to an empty dictionary.
        """
        self.name = name
        self.type = type
        self.description = description
        self.config = config if config is not None else {}

    def get_meta(self) -> dict:
        """
        Retrieves metadata about the component, including its name, type, description, and configuration.

        Returns:
            dict: A dictionary containing the component's metadata.
        """
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "config": self.config,
        }

    def update_config(self, key: str, value):
        """
        Updates a configuration parameter for the component.

        Args:
            key (str): The configuration parameter's key.
            value: The new value for the configuration parameter.
        """
        self.config[key] = value

    def __repr__(self):
        """
        Provides a string representation of the component for debugging and logging purposes.

        Returns:
            str: A string representation of the OpeaComponent instance.
        """
        return f"OpeaComponent(name={self.name}, type={self.type}, description={self.description})"

