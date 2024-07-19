from ..base_agent import BaseAgent
from ...utils import has_multi_tool_inputs, tool_renderer
from langchain.agents import AgentExecutor, create_react_agent
from .prompt import hwchase17_react_prompt

class ReActAgentwithLangchain(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        prompt = hwchase17_react_prompt
        if has_multi_tool_inputs(self.tools_descriptions):
            raise ValueError("Only supports single input tools when using strategy == react")
        else:
            agent_chain = create_react_agent(
                self.llm_endpoint, self.tools_descriptions, prompt, tools_renderer=tool_renderer
            )
        self.app = AgentExecutor(
            agent=agent_chain, tools=self.tools_descriptions, verbose=True, handle_parsing_errors=True
        )
