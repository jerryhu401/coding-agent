import os

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types  # avoids shadowing built-in types module

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from agent.adk_agent import build_agent
from agent.tools import make_tools


class AdkAgent(BaseAgent):

    @staticmethod
    def name() -> str:
        return "adk_agent"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        # self.model_name lets the --model CLI flag override the default
        model = self.model_name or "openrouter/anthropic/claude-opus-4-6"

        tools = make_tools(environment)
        adk_agent = build_agent(model=model, api_key=api_key, tools=tools)

        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name="agent_challenge", user_id="user", session_id="session"
        )

        runner = Runner(
            agent=adk_agent,
            app_name="agent_challenge",
            session_service=session_service,
        )

        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=instruction)],
        )

        # runner.run_async drives the ReAct loop; we break on the final response
        async for event in runner.run_async(
            user_id="user",
            session_id="session",
            new_message=content,
        ):
            if event.is_final_response():
                break
