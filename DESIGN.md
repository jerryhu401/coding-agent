# Agent Design Process

## Milestone 1: Understanding the Task and Requirements

Read documentation on ADK, LiteLLM, terminal-bench/harbor, and agentic design in general to understand what we're building and how the pieces fit together.

## Milestone 2: Setting Up the Environment

Install all required packages and dependencies and set up the environment to run in:
- ADK, LiteLLM, terminal-bench, harbor
- Run in WSL (required for harbor's Docker integration)

Ran preliminary tests to make sure packages and API connections work before writing the agent.
*took way longer than it should*

## Milestone 3: Skeleton Code and End-to-End Connection Test ##

Three steps:
1. Learn to set up and invoke the basic API calls using ADK and LiteLLM — test with a simple "say hello world" prompt
2. Basic tool use test: write a `run_shell` command, hook it up to LiteLLM, and see if Claude can do something simple like "run ls and tell me what's in the directory"
3. Set up Docker + harbor end-to-end

## Milestone 4: Bare Bones Simple Agent

Combine steps from Milestone 3 into a working agent:
- Inherit from `BaseAgent` from harbor and plug in our agent
- Simple system prompt: "you are an independent coding agent solving tasks within a docker container"
- Only `run_shell` tool provided
- this was already surprisingly capable, able to somewhat consistently solve several of the target tasks

## Milestone 5: Optimizations (Iterating on Test Failures)

**`write_file` tool**
Early failures on mangled Python scripts due to writing files purely with shell commands. Added a `write_file` tool for more accurate and efficient file writing. Implementation: the Python script opens a file (which creates it on the host), then uses a harbor library call to upload it into the container.

**Improved `run_shell`**
Updated the tool to also return the exit code, and truncate output to avoid blowing up the context window.

**`read_file` tool**
Failures on tasks requiring efficient and accurate reads of files with large contents — clunky shell script reads, risk of mangling, and potential truncation issues. Added an explicit `read_file` tool, same upload/download implementation as `write_file`.

**`read_image` tool**
Failure on the gcode decoding task (timeout and incorrect result). Fix was a `read_image` tool.

Manually walked through solving the problem with Claude: the agent could process gcode into a readable format and easily create the corresponding image with the toolpath rendered on it, but struggled to extract the flag from the image via shell commands. But from there I was able to very easily just read the image and see the flag. The solution: just read the image directly with a vision model. `read_image` makes a separate LiteLLM call to a vision model (Claude again) and returns the output. But this can be changed to use more specific vision models. I think this vision model idea is pretty powerful for opening up a fundamental new capability for the agent(but this has complications as seen later)

**Structured image insight**
Interestingly, adding `read_image` caused the agent to repeatedly fail on the chess-next-move task. The reason: the chessboard is provided as a PNG with basically no information that directly encodes the board state. Without `read_image`, the agent writes a script to parse the Unicode in the image metadata and reconstruct the board — deterministic and accurate. But when the agent sees `read_image`, it prioritizes the tool, which leads to misinterpretation of the board by the vision model and failure.

Fix: added a line in the system prompt asking the agent to prefer deterministic programmatic image analysis over vision model usage when the image is known to be structured and organized (like grids or a chessboard — e.g., use PIL to sample pixel colors at known positions or do template matching).

Interesting insight here: I almost went down the rabbit hole of optimizing the `read_image` tool (splitting into sub-images, validating pieces by subset and recombining), but there was a much more elegant solution at the prompt level.

**Logging-based debugging**
Early on, failed runs were quite hard to reason about. Direct run time errors and exceptions were easier to fix, but sometimes it was still unclear how the agent got there, and incorrect tests that ran fine was even harder to debug. The task scored 0, but it wasn't clear whether the agent misunderstood the task, hit a tool error, ran out of time mid-execution, or made a logically wrong choice. A couple times I basically guessed what the agent did and tried to tune the prompt or scavanged for clues in the result. 

Adding event logging allowed me to see what the agent did, what it got back, and what it then decided to do. A clear example: the db-wal-recovery task kept timing out. Without the trajectory it looked like a general timeout issue — maybe the task was just too slow. Reading the log showed the agent ran `sqlite3` on the database in its very first action, which caused SQLite to silently discard the XOR-encrypted WAL file permanently(at least this is the conclusion I came to with little to no idea on what these things actually are...). From that point the data was unrecoverable, but the agent didn't know that — it spent the entire remaining budget (198 tool calls) trying increasingly desperate recovery approaches on data that no longer existed. The root cause was a single wrong first action, and the fix was just tuning the prompt. 

The logging also helped me find the above mentioned chess image reading issue, which would have been near impossible to realize without explicit understanding of what the agent chose to do. 

**Prompt tuning**
Further prompt refinements, kept as general as possible — giving strong direct hints only on target tasks would clearly get good performance but wouldn't generalize:
- Explore the directory before doing anything
- Check if tests are provided in the container; if so, always read them first and run them after completing the implementation before signaling done
- Note: some tasks time out only after the agent runs tests post-implementation, but the harbor verifier still treats this as a pass as long as the solution is correct
- Closely follow task requirements and hints; always use packages that are suggested rather than similar alternatives that may give different results. This was seen when the agent used a different library for mteb-retrieve that produced different results to the one specified in the task
- Preserve directory structure (tests may check this)
- Remove unneeded build artifacts before submission (e.g., object files)

## Next Steps ##
- Lots of the prompt changes include examples that are still pretty related to specific tasks, I would like to move to even more general and broadly applicable tools
- Testing with a smaller but faster model can yield different results. It would likely need more guidance and perhaps specific tools, but running faster also means more interactions can happen in the provided time range, and it is technically lower cost
- overall, there is a lot that can still be optimized. I was surprised at the relative simplicity of the harness compared to the capability of the agent, it really shows the power of combining the reasoning capabilities of a strong model with the core think -> tool -> observe -> think loop. 