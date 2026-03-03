import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types  # avoids shadowing built-in types module

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from agent.adk_agent import build_agent
from agent.tools import make_tools


def _log_event(log_file, event) -> None:
    """Append a single ADK event as a JSON line to the trajectory log."""
    entry = {"author": event.author}
    if event.content and event.content.parts:
        parts = []
        for part in event.content.parts:
            if part.text:
                parts.append({"text": part.text})
            elif part.function_call:
                parts.append({"tool_call": {"name": part.function_call.name, "args": dict(part.function_call.args)}})
            elif part.function_response:
                resp = part.function_response.response
                parts.append({"tool_response": {"name": part.function_response.name, "output": str(resp)}})
        entry["parts"] = parts
    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()


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

        tools = make_tools(environment, api_key=api_key, model=model)
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

        # Open trajectory log in the harbor agent/ directory
        log_path = Path(self.logs_dir) / "trajectory.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log_file:
            # runner.run_async drives the ReAct loop; we break on the final response
            async for event in runner.run_async(
                user_id="user",
                session_id="session",
                new_message=content,
            ):
                _log_event(log_file, event)
                if event.is_final_response():
                    break
