"""SUPER Expert — OpenReward sandbox environment for scientific code execution.

Agents must execute tasks from research GitHub repositories: clone repos, install
dependencies, run code (often ML training), and report specific results.

Paper: https://arxiv.org/abs/2409.07440
Dataset: https://huggingface.co/datasets/allenai/super
"""

import json
import logging
import os
from pathlib import Path

from openreward import AsyncOpenReward, SandboxSettings
from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool
from pydantic import BaseModel

from evaluate import evaluate, parse_answer

logger = logging.getLogger(__name__)

# --- Module-level data loading ---

if os.path.exists("/orwd_data"):
    _DATA_DIR = Path("/orwd_data")
else:
    _DATA_DIR = Path(__file__).parent

_all_records: dict[str, dict] = {}
_tasks: list[JSONObject] = []

_json_path = _DATA_DIR / "tasks_test.json"
if not _json_path.exists():
    logger.warning(f"Data file not found: {_json_path}")
else:
    with open(_json_path) as _f:
        _records = json.load(_f)
    for _record in _records:
        _record_id = _record["id"]
        _all_records[_record_id] = _record
        _tasks.append({
            "id": _record_id,
            "task_id_short": _record["task_id_short"],
        })


# --- Pydantic parameter models ---

class BashParams(BaseModel, extra="forbid"):
    command: str


class SubmitParams(BaseModel, extra="forbid"):
    """Submit the answer as a JSON value."""
    answer: str


# --- Environment class ---

class SUPERExpert(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)

        record_id = str(task_spec["id"])
        if record_id not in _all_records:
            raise ValueError(f"Unknown task id: {record_id}")

        record = _all_records[record_id]
        self.query: str = record["query"]
        self.gold_answer = record["answer"]
        self.landmarks: list[str] = record.get("landmarks", [])

        api_key = (
            secrets.get("OPENREWARD_API_KEY")
            or secrets.get("api_key")
            or os.environ.get("OPENREWARD_API_KEY", "").strip('"')
        )
        if not api_key:
            raise ValueError("OpenReward API key required (pass as OPENREWARD_API_KEY)")

        self.sandbox_settings = SandboxSettings(
            environment="GeneralReasoning/SUPER-Expert",
            image="generalreasoning/python-ds:3.12-tools",
            machine_size="2:4",
            block_network=False,
        )

        or_client = AsyncOpenReward(api_key=api_key)
        self.sandbox = or_client.sandbox(self.sandbox_settings)

        self.submitted = False

    async def setup(self) -> None:
        await self.sandbox.start()

    async def teardown(self) -> None:
        await self.sandbox.stop()

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "test":
            return _tasks
        return []

    async def get_prompt(self) -> list[TextBlock]:
        prompt = f"""{self.query}

## Environment

You have access to a Linux sandbox with Python 3.12, common ML/data science libraries,
and network access. Use the `bash` tool to clone repositories, install dependencies,
run code, and produce your answer.

When you have the answer, use the `submit` tool with a JSON string containing your result.
You get one submission attempt."""

        return [TextBlock(text=prompt)]

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """Execute a bash command in the sandbox environment."""
        result = await self.sandbox.run(params.command.strip())
        output, code = result

        if result.truncated:
            output = f"...(truncated, output exceeded limit)\n{output}"

        return ToolOutput(
            blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
            metadata={"output": output, "exit_code": code, "truncated": result.truncated},
            reward=0.0,
            finished=False,
        )

    @tool
    async def submit(self, params: SubmitParams) -> ToolOutput:
        """Submit your answer for evaluation.

        The answer should be a JSON string matching the expected output format
        described in the task. This is a terminal action — one attempt only.
        """
        if self.submitted:
            return ToolOutput(
                blocks=[TextBlock(text="Already submitted. Only one submission is allowed.")],
                metadata={"error": "already_submitted"},
                reward=0.0,
                finished=True,
            )

        self.submitted = True

        # Parse the submitted answer
        predicted = parse_answer(params.answer)

        # Evaluate against gold
        try:
            score = evaluate(predicted, self.gold_answer)
        except Exception as e:
            logger.exception("Evaluation error")
            return ToolOutput(
                blocks=[TextBlock(text=f"Evaluation error: {e}")],
                metadata={"error": str(e)},
                reward=0.0,
                finished=True,
            )

        result_text = f"""Submission Results:
- Output Match: {score:.4f}
- Reward: {score:.4f}"""

        return ToolOutput(
            blocks=[TextBlock(text=result_text)],
            metadata={
                "output_match": score,
            },
            reward=score,
            finished=True,
        )


if __name__ == "__main__":
    from openreward.environments import Server
    Server([SUPERExpert]).run()
