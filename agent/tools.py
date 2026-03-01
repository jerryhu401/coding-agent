import base64
import os
import tempfile

import litellm
from harbor.environments.base import BaseEnvironment

_TRUNCATE_CHARS = 8000


def make_tools(environment: BaseEnvironment, api_key: str, model: str) -> list:
    #call this to get a list of tools usable by agent

    async def run_bash(command: str) -> str:
        """
        Run a shell command in the Docker container. Returns stdout+stderr and exit code.
        compared to last version, added the exit code on top of stdout and err
        """
        try:
            result = await environment.exec(command, timeout_sec=300)
        except RuntimeError as e:
            # harbor raises RuntimeError when the command exceeds timeout_sec;
            # return an error string so the agent can adapt instead of crashing the trial
            return f"[error] command timed out: {e}"
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr
        if len(output) > _TRUNCATE_CHARS:
            output = f"[truncated, showing last {_TRUNCATE_CHARS} chars]\n" + output[-_TRUNCATE_CHARS:]
        return f"[exit {result.return_code}]\n{output}" if output else f"[exit {result.return_code}]"

    async def write_file(path: str, content: str) -> str:
        """
        Write content to a file at the given path in the Docker container.
        Uses a temp file on the host + upload_file to avoid shell quoting issues.
        the temp file approach is because we need to write into the container,
        directly using python's open would go onto the host machine(I could be wrong here)
        """
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tmp") as f:
            f.write(content)
            tmp_path = f.name
        try:
            #take advantage of what harbor provides(in BaseEnvironment I think)
            await environment.upload_file(tmp_path, path)
            return "[exit 0]"
        except Exception as e:
            return f"[error] {e}"
        finally:
            os.unlink(tmp_path)
            #remove temp file, we opened with delete=false to have time for upload

    async def read_file(path: str) -> str:
        """
        Read a file from the Docker container and return its contents.
        Uses download_file to copy from container to host — avoids shell output
        truncation that affects run_bash when files are large.
        """
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tmp") as f:
            tmp_path = f.name
        try:
            await environment.download_file(path, tmp_path)
            with open(tmp_path, "r", errors="replace") as f:
                content = f.read()
            if len(content) > _TRUNCATE_CHARS:
                content = f"[truncated, showing first {_TRUNCATE_CHARS} chars]\n" + content[:_TRUNCATE_CHARS]
            return content
        except Exception as e:
            return f"[error] {e}"
        finally:
            os.unlink(tmp_path)

    async def read_image(path: str, prompt: str) -> str:
        """
        Read an image file from the Docker container and use a vision model to interpret it.
        Useful for understanding visual output such as plots, rendered text, or diagrams.
        Returns the model's description of the image contents.
        """
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".png") as f:
            tmp_path = f.name
        try:
            await environment.download_file(path, tmp_path)
            with open(tmp_path, "rb") as f:
                img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
            response = await litellm.acompletion(
                model=model,
                api_key=api_key,
                api_base="https://openrouter.ai/api/v1",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[error] {e}"
        finally:
            os.unlink(tmp_path)

    return [run_bash, write_file, read_file, read_image]
