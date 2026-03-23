"""Download and prepare SUPER Expert data for the OpenReward environment.

Downloads the allenai/super Expert split from HuggingFace and saves task specs.
GitHub repos referenced in tasks are cloned by agents at runtime (not pre-staged).

Usage:
    uv run python prepare_data.py
"""

import json
from hashlib import md5
from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path(__file__).parent


def main():
    print("=== SUPER Expert Data Preparation ===\n")

    print("Downloading allenai/super Expert split...")
    ds = load_dataset("allenai/super", "Expert")
    samples = list(map(dict, ds["all_examples"]))
    print(f"  {len(samples)} tasks")

    tasks = []
    for obj in samples:
        instance_rep = md5(str(obj["query_components"]).encode("utf-8")).hexdigest()
        task_id = f"{obj['task_id']}_{instance_rep}"

        # Parse gold answer from JSON string if needed
        answer = obj["answer"]
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                pass  # Keep as string

        tasks.append({
            "id": task_id,
            "task_id_short": obj["task_id"],
            "query": obj["query"],
            "answer": answer,
            "landmarks": obj.get("landmarks", []),
            "github_repo": obj.get("github_repo", ""),
        })

    # Sort by id for stable ordering
    tasks.sort(key=lambda t: t["id"])

    with open(OUTPUT_DIR / "tasks_test.json", "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"\nDone!")
    print(f"  tasks_test.json: {len(tasks)} tasks")

    # Summary of answer types
    from collections import Counter
    answer_types = Counter(type(t["answer"]).__name__ for t in tasks)
    print(f"  Answer types: {answer_types}")


if __name__ == "__main__":
    main()
