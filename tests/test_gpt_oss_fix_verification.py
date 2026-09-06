from notebooks import gpt_oss_fix_verification as fix


def test_fix_profiles_are_exactly_the_bounded_two_by_three_verification_matrix():
    assert [profile["name"] for profile in fix.FIX_PROFILES] == [
        "reasoning_low_same_budget_32",
        "reasoning_low_larger_budget_256",
        "no_reasoning_larger_budget_256",
    ]
    assert len(fix.GPT_OSS_MODELS) * len(fix.FIX_PROFILES) == 6


def test_fix_sanitizer_returns_only_fixed_safe_fields_without_error_text():
    class FakeInvalidRequest(RuntimeError):
        status_code = 400
        body = {
            "error": {
                "type": "invalid_request_error",
                "code": "json_validate_failed",
                "param": None,
                "message": "provider text must never be retained",
            }
        }

    record = fix.sanitize_fix_failure(
        model="openai/gpt-oss-20b",
        profile="reasoning_low_same_budget_32",
        error=FakeInvalidRequest(),
    )

    assert record == {
        "model": "openai/gpt-oss-20b",
        "profile": "reasoning_low_same_budget_32",
        "status": "fail",
        "http_status": 400,
        "provider_type": "invalid_request_error",
        "code": "json_validate_failed",
        "parameter": None,
        "category": "invalid_request",
    }
    assert "provider text" not in repr(record)


def test_fix_success_record_has_no_error_fields():
    record = fix._success_record("openai/gpt-oss-120b", "no_reasoning_larger_budget_256")
    assert record == {
        "model": "openai/gpt-oss-120b",
        "profile": "no_reasoning_larger_budget_256",
        "status": "pass",
        "http_status": None,
        "provider_type": None,
        "code": None,
        "parameter": None,
        "category": "ok",
    }
