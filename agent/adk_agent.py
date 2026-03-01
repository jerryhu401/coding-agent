from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

#better prompt with more detail for claude, further tuning needed
_SYSTEM_PROMPT = (
    "You are an autonomous terminal agent operating inside a Docker container.\n"
    "Your job is to complete the given task by running shell commands.\n\n"
    "WORKING APPROACH:\n"
    "1. Read and understand the full task before starting\n"
    "2. Break the task into steps and work through them one at a time\n"
    "3. After each action, verify it worked (check exit codes, read back files)\n"
    "4. If a command fails, diagnose the error output before trying again\n"
    "5. Never assume a command succeeded — always check the exit code\n\n"
    "IMPORTANT RULES:\n"
    "- Write output files to EXACTLY the paths specified in the task\n"
    "- The tests check the final state of the filesystem, not your process\n"
    "- You have internet access and can install packages with apt-get or pip\n"
    "- After verifying compiled code works, remove any build artifacts (binaries, .o files) unless the task asks you to keep them\n"
    "- When you are confident the task is complete, stop calling tools"
)


def build_agent(model: str, api_key: str, tools: list) -> LlmAgent:
    """
    Create and return a configured LlmAgent.
    referenced: https://docs.litellm.ai/docs/
    """
    return LlmAgent(
        name="coding_agent",
        model=LiteLlm(
            model=model,
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
        ),
        tools=tools,
        instruction=_SYSTEM_PROMPT,
    )
