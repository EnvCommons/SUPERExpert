"""Unit tests for the SUPER Expert OpenReward environment."""

import json
from pathlib import Path

import pytest

from evaluate import evaluate, parse_answer


# --- Scoring tests ---


class TestEvaluate:
    """Tests for type-aware exact match evaluation."""

    def test_float_exact(self):
        assert evaluate(0.95, 0.95) == 1.0

    def test_float_within_epsilon(self):
        assert evaluate(0.951, 0.95) == 1.0

    def test_float_outside_epsilon(self):
        assert evaluate(0.97, 0.95) == 0.0

    def test_string_exact(self):
        assert evaluate("hello", "hello") == 1.0

    def test_string_strip(self):
        assert evaluate("  hello  ", "hello") == 1.0

    def test_string_mismatch(self):
        assert evaluate("hello", "world") == 0.0

    def test_int_as_float(self):
        # int and float should be comparable
        assert evaluate(1, 1.0) == 1.0

    def test_dict_all_match(self):
        gold = {"a": 1.0, "b": 2.0}
        pred = {"a": 1.0, "b": 2.0}
        assert evaluate(pred, gold) == 1.0

    def test_dict_partial_match(self):
        gold = {"a": 1.0, "b": 2.0}
        pred = {"a": 1.0, "b": 9.0}
        assert evaluate(pred, gold) == 0.5

    def test_dict_missing_key(self):
        gold = {"a": 1.0, "b": 2.0}
        pred = {"a": 1.0}  # missing "b"
        assert evaluate(pred, gold) == 0.5

    def test_list_all_match(self):
        assert evaluate([1.0, 2.0], [1.0, 2.0]) == 1.0

    def test_list_partial(self):
        assert evaluate([1.0, 9.0], [1.0, 2.0]) == 0.5

    def test_type_mismatch(self):
        assert evaluate("hello", 1.0) == 0.0

    def test_none_predicted(self):
        assert evaluate(None, 1.0) == 0.0


class TestParseAnswer:
    def test_json_dict(self):
        result = parse_answer('{"a": 1}')
        assert result == {"a": 1}

    def test_json_list(self):
        result = parse_answer('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_plain_string(self):
        result = parse_answer("hello")
        assert result == "hello"

    def test_already_dict(self):
        result = parse_answer({"a": 1})
        assert result == {"a": 1}


# --- Task structure tests ---


class TestTaskStructure:
    @pytest.fixture(autouse=True)
    def _load_data(self):
        path = Path(__file__).parent / "tasks_test.json"
        if not path.exists():
            pytest.skip("Data not found — run prepare_data.py first")
        with open(path) as f:
            self.tasks = json.load(f)

    def test_count(self):
        assert len(self.tasks) == 45

    def test_required_fields(self):
        required = {"id", "task_id_short", "query", "answer"}
        for task in self.tasks:
            assert required.issubset(set(task.keys())), f"Missing fields in {task['id']}"

    def test_all_have_answers(self):
        for task in self.tasks:
            assert task["answer"] is not None, f"No answer for {task['id']}"

    def test_stable_ordering(self):
        ids = [t["id"] for t in self.tasks]
        assert ids == sorted(ids)

    def test_unique_ids(self):
        ids = [t["id"] for t in self.tasks]
        assert len(ids) == len(set(ids))


# --- Environment class tests ---


class TestSUPERExpertEnv:
    @pytest.fixture(autouse=True)
    def _check_data(self):
        if not (Path(__file__).parent / "tasks_test.json").exists():
            pytest.skip("Data not found — run prepare_data.py first")

    def test_list_splits(self):
        from super_expert import SUPERExpert
        assert SUPERExpert.list_splits() == ["test"]

    def test_list_tasks(self):
        from super_expert import SUPERExpert
        tasks = SUPERExpert.list_tasks("test")
        assert len(tasks) == 45

    def test_task_spec_fields(self):
        from super_expert import SUPERExpert
        tasks = SUPERExpert.list_tasks("test")
        for task in tasks[:5]:
            assert "id" in task
            assert "task_id_short" in task
            # Should NOT expose gold answers
            assert "answer" not in task
            assert "query" not in task
