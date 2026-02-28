"""
Make an LiteLLM agent, then try to have it use a "tool"
ls and tell me whats in the dir

Using this to learn the basics of interacting with adk, comments are notes to 
myself as I try and figure out how the work loop works and what the data structures
are. 

python3.12 test_adk.py
"""
import asyncio
import os
import subprocess
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

load_dotenv()

key = os.getenv("OPENROUTER_API_KEY")
if not key:
    raise SystemExit("ERROR: Set OPENROUTER_API_KEY in .env first")


def run_bash(command: str) -> str:
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True
    )
    return result.stdout + result.stderr


agent = LlmAgent(
    name="test_agent",
    model=LiteLlm(
        model="openrouter/anthropic/claude-opus-4-6",
        api_key=key,
        api_base="https://openrouter.ai/api/v1",
    ),
    tools=[run_bash],
    instruction="You are a helpful assistant. Use run_bash to answer questions.",
)


async def main():
    session_service = InMemorySessionService()
    '''
    session is the history of the loop that goes back to claude, in memory as
    it is on RAM not disk. This lets the LLM remember what happened in each step
    '''
    await session_service.create_session(
        app_name="test", user_id="user", session_id="session"
    )
    runner = Runner(agent=agent, app_name="test", session_service=session_service)
    '''
    runner is what runs the loop, sends prompt and history(session) to LLM,
    LLM responds, runner sees and executes some command that LLM chooses, then
    repeats this cycle until final response. It runs async so network calls with
    LLM does not block. Also enforces turn limit?(not sure where to set this).
    Also routes responses to the rigth "tools". In this case just run_bash
    '''
    
    print("Prompt: 'Run ls and tell me what files are here'\n")

    '''
    event is just when something happens on a turn(sending message to LLM and got
    response, running a tool, LLM then replied, etc)
    '''

    '''
    Content is something gemini has and I guess used for adk as well? Basically,
    it is one message in the conversation history. Can be from user, LLM, or 
    a tool. 

    Part is also an internal adk type, can be things like text, function_call,
    function_response and others
    '''
    async for event in runner.run_async(
        user_id="user",
        session_id="session",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="Run ls and tell me what files are here")]
        ),
    ):
        '''
        final response checks if LLM stopped using tools
        '''
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent reply:\n{part.text}")


asyncio.run(main())
