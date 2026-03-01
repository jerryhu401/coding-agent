import os
import tempfile

from harbor.environments.base import BaseEnvironment

_TRUNCATE_CHARS = 8000


def make_tools(environment: BaseEnvironment) -> list:
    #call this to get a list of tools usable by agent

    async def run_bash(command: str) -> str:
        """
        Run a shell command in the Docker container. Returns stdout+stderr and exit code.
        compared to last version, added the exit code on top of stdout and err
        """
        result = await environment.exec(command, timeout_sec=120)
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

    return [run_bash, write_file]
