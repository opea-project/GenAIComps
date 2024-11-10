import json

import uuid
from datetime import datetime

from typing import List, Optional

from pydantic import BaseModel

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph


class PersistenceConfig(BaseModel):
    checkpointer: bool = False
    store: bool = False

class PersistenceInfo(BaseModel):
    user_id: str = None
    thread_id: str = None
    started_at: datetime


class AgentPersistence:
    def __init__(self, config: PersistenceConfig):
        # for short-term memory
        self.checkpointer = None
        # for long-term memory
        self.store = None
        self.config = config
        print(f"Initializing AgentPersistence: {config}")
        self.initialize()

    def initialize(self) -> None:
        if self.config.checkpointer:
            self.checkpointer = MemorySaver()
        if self.config.store:
            self.store = InMemoryStore()

    def get_state_history(self, config, graph: StateGraph):
        pass

    def get_state(self, config, graph: StateGraph):
        pass

    def update_state(self, config, graph: StateGraph):
        pass
