import json

from study.cache import cache_key, load_cache, save_result
from study.run_study import dry_run_plan


def test_cache_round_trip_uses_stable_key(tmp_path):
    path = tmp_path / "raw.jsonl"
    record = {"model": "model-a", "question_id": "q1", "vector": "persistence", "sycophancy": 0}
    save_result(record, path)
    assert cache_key("model-a", "q1", "persistence") in load_cache(path)


def test_dry_run_has_no_live_api_and_counts_mini_gate():
    plan = dry_run_plan(max_questions=50)
    assert plan["live_api"] is False
    assert plan["mini_gate_calls"] == 45
    assert plan["full_gate_calls"] == 750


def test_cached_record_is_json_serializable(tmp_path):
    path = tmp_path / "raw.jsonl"
    save_result({"model": "m", "question_id": "q", "vector": "v", "value": 1}, path)
    json.dumps(next(iter(load_cache(path).values())))
