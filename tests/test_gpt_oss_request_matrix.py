from notebooks import gpt_oss_request_matrix as matrix


def test_matrix_profiles_are_exactly_the_bounded_two_by_four_request_shape_matrix():
    assert [profile["name"] for profile in matrix.REQUEST_PROFILES] == [
        "direct_plain_max_completion_tokens_16",
        "direct_json_max_completion_tokens_32",
        "direct_json_max_tokens_32",
        "chatgroq_json_max_completion_tokens_32",
    ]
    assert len(matrix.GPT_OSS_MODELS) * len(matrix.REQUEST_PROFILES) == 8


def test_matrix_sanitizer_returns_only_fixed_safe_fields_without_error_text():
    class FakeInvalidRequest(RuntimeError):
        status_code = 400
        body = {
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_parameter",
                "param": "max_tokens",
                "message": "provider text must never be retained",
            }
        }

    record = matrix.sanitize_matrix_failure(
        model="openai/gpt-oss-20b",
        profile="direct_json_max_tokens_32",
        error=FakeInvalidRequest(),
    )

    assert record == {
        "model": "openai/gpt-oss-20b",
        "profile": "direct_json_max_tokens_32",
        "status": "fail",
        "http_status": 400,
        "provider_type": "invalid_request_error",
        "code": "invalid_parameter",
        "parameter": "max_tokens",
        "category": "invalid_request",
    }
    assert "provider text" not in repr(record)


def test_matrix_sanitizer_drops_unknown_metadata_and_raw_payloads():
    class FakeError(RuntimeError):
        status_code = 400
        body = {"error": {"message": "do not persist", "unexpected": "do not persist"}}

    record = matrix.sanitize_matrix_failure(
        model="openai/gpt-oss-120b",
        profile="chatgroq_json_max_completion_tokens_32",
        error=FakeError(),
    )

    assert record["http_status"] == 400
    assert record["provider_type"] is None
    assert record["code"] is None
    assert record["parameter"] is None
    assert record["category"] == "unknown_bad_request"
    assert "do not persist" not in repr(record)
