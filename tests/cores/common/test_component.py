# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from comps import OpeaComponent, OpeaComponentRegistry, OpeaComponentLoader


class TestOpeaComponent(unittest.TestCase):
    class MockOpeaComponent(OpeaComponent):
        def check_health(self) -> bool:
            return True

        async def invoke(self, *args, **kwargs):
            return "Service accessed"

    def test_initialization(self):
        component = self.MockOpeaComponent("TestComponent", "embedding", "Test description")

        self.assertEqual(component.name, "TestComponent")
        self.assertEqual(component.type, "embedding")
        self.assertEqual(component.description, "Test description")
        self.assertEqual(component.config, {})

    def test_get_meta(self):
        component = self.MockOpeaComponent("TestComponent", "embedding", "Test description", {"key": "value"})
        meta = component.get_meta()

        self.assertEqual(meta["name"], "TestComponent")
        self.assertEqual(meta["type"], "embedding")
        self.assertEqual(meta["description"], "Test description")
        self.assertEqual(meta["config"], {"key": "value"})

    def test_update_config(self):
        component = self.MockOpeaComponent("TestComponent", "embedding", "Test description")
        component.update_config("key", "new_value")

        self.assertEqual(component.config["key"], "new_value")


class TestOpeaComponentRegistry(unittest.TestCase):
    def test_register_component(self):
        # Create a mock component class
        class MockComponent(OpeaComponent):
            def check_health(self) -> bool:
                return True

            async def invoke(self, *args, **kwargs):
                return "Service accessed"

        # Register the component
        OpeaComponentRegistry.register("MockComponent")(MockComponent)

        # Retrieve the component and ensure it's correct
        retrieved_component_class = OpeaComponentRegistry.get("MockComponent")
        self.assertEqual(retrieved_component_class, MockComponent)

        # Test exception for already registered component
        with self.assertRaises(ValueError):
            OpeaComponentRegistry.register("MockComponent")(MockComponent)

    def test_unregister_component(self):
        # Create a mock component class
        class MockComponent(OpeaComponent):
            def check_health(self) -> bool:
                return True

            async def invoke(self, *args, **kwargs):
                return "Service accessed"

        # Register and then unregister the component
        OpeaComponentRegistry.register("MockComponent")(MockComponent)
        OpeaComponentRegistry.unregister("MockComponent")

        # Ensure the component is no longer in the registry
        with self.assertRaises(KeyError):
            OpeaComponentRegistry.get("MockComponent")


class TestOpeaComponentLoader(unittest.TestCase):
    def test_invoke_registered_component(self):
        # Mock a component with the invoke method
        class MockComponent(OpeaComponent):
            def check_health(self) -> bool:
                return True

            async def invoke(self, *args, **kwargs):
                return "Service accessed"

        # Register the mock component
        OpeaComponentRegistry.register("MockComponent")(MockComponent)

        # Create loader for the component
        loader = OpeaComponentLoader("MockComponent")
        
        # Invoke the component
        result = asyncio.run(loader.invoke("arg1", key="value"))

        # Check the result
        self.assertEqual(result, "Service accessed")

    def test_invoke_unregistered_component(self):
        # Attempt to load a component that is not registered
        with self.assertRaises(KeyError):
            OpeaComponentLoader("UnregisteredComponent")

if __name__ == "__main__":
    unittest.main()

