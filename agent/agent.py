import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types #to not conflict with python

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class AdkAgent(BaseAgent):
    #Referencing https://harborframework.com/docs/agents
    
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
        model = self.model_name or "openrouter/anthropic/claude-opus-4-6"

        async def run_bash(command: str) -> str:
            """Run a shell command in the Docker container and return its output."""
            result = await environment.exec(command, timeout_sec=60)
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += result.stderr
            return output or f"(exit code: {result.return_code})"

        adk_agent = LlmAgent(
            name="coding_agent",
            model=LiteLlm(
                model=model,
                api_key=api_key,
                api_base="https://openrouter.ai/api/v1",
            ),
            tools=[run_bash],
            instruction=(
                "You are a coding agent working in a Docker container. "
                "Use run_bash to execute shell commands and complete the given task. "
                "Be systematic: explore the environment, understand the requirements, "
                "implement your solution, and verify your work."
            ),
        )

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

        '''
        runs loops, each being an event. Uses session as memory and new_message
        is the first message/event sent to the LLM
        '''
        async for event in runner.run_async(
            user_id="user",
            session_id="session",
            new_message=content,
        ):
            if event.is_final_response():
                break
