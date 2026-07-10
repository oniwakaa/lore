# tests/test_json_utils.py
"""Tests for shared JSON parsing utilities."""
import pytest
from lore.json_utils import (
    strip_fences,
    extract_json_object,
    repair_truncated_json,
    parse_json_response,
)


class TestStripFences:
    def test_strips_json_fences(self):
        text = '```json\n{"key": "value"}\n```'
        assert strip_fences(text) == '{"key": "value"}'

    def test_strips_bare_fences(self):
        text = '```\n{"key": "value"}\n```'
        assert strip_fences(text) == '{"key": "value"}'

    def test_no_fences_unchanged(self):
        text = '{"key": "value"}'
        assert strip_fences(text) == '{"key": "value"}'

    def test_strips_fences_with_trailing_whitespace(self):
        text = '```json\n{"key": "value"}\n```\n\n  '
        assert strip_fences(text) == '{"key": "value"}'


class TestExtractJsonObject:
    def test_extracts_from_surrounding_text(self):
        text = 'Here is the plan:\n{"subtasks": []}\nDone.'
        assert extract_json_object(text) == '{"subtasks": []}'

    def test_returns_none_if_no_json(self):
        assert extract_json_object("no json here") is None

    def test_extracts_nested_objects(self):
        text = 'Result: {"a": {"b": 1}, "c": [1, 2]}'
        result = extract_json_object(text)
        assert result is not None
        assert '"a"' in result and '"b"' in result

    def test_extracts_first_object(self):
        text = '{"first": 1} and {"second": 2}'
        result = extract_json_object(text)
        assert result is not None
        assert '"first"' in result


class TestRepairTruncatedJson:
    def test_repairs_missing_brace(self):
        truncated = '{"key": "value"'
        result = repair_truncated_json(truncated)
        assert result == {"key": "value"}

    def test_repairs_missing_bracket_and_brace(self):
        truncated = '{"items": [1, 2, 3'
        result = repair_truncated_json(truncated)
        assert result == {"items": [1, 2, 3]}

    def test_repairs_mid_string(self):
        truncated = '{"key": "truncated val'
        result = repair_truncated_json(truncated)
        assert result is not None
        assert "key" in result

    def test_returns_none_for_unparseable(self):
        result = repair_truncated_json("not json at all")
        assert result is None

    def test_preserves_nested_structure(self):
        truncated = '{"a": {"b": [1, 2'
        result = repair_truncated_json(truncated)
        assert result is not None
        assert result["a"]["b"] == [1, 2]


class TestParseJsonResponse:
    def test_parses_clean_json(self):
        assert parse_json_response('{"key": "value"}') == {"key": "value"}

    def test_parses_fenced_json(self):
        raw = '```json\n{"key": "value"}\n```'
        assert parse_json_response(raw) == {"key": "value"}

    def test_parses_json_with_surrounding_text(self):
        raw = 'Here is the plan:\n{"subtasks": []}\nDone.'
        assert parse_json_response(raw) == {"subtasks": []}

    def test_fixes_trailing_commas(self):
        raw = '{"key": "value",}'
        assert parse_json_response(raw) == {"key": "value"}

    def test_repairs_truncated(self):
        raw = '{"subtasks": [{"id": "s1"'
        result = parse_json_response(raw)
        assert result is not None
        assert "subtasks" in result

    def test_returns_none_for_empty_string(self):
        assert parse_json_response("") is None

    def test_returns_none_for_no_json(self):
        assert parse_json_response("just some text") is None

    def test_parses_nested_objects(self):
        raw = '{"outer": {"inner": {"deep": [1, 2, 3]}}}'
        assert parse_json_response(raw) == {"outer": {"inner": {"deep": [1, 2, 3]}}}

    def test_parses_with_trailing_commas_in_nested(self):
        raw = '{"a": [1, 2,], "b": {"c": 1,},}'
        result = parse_json_response(raw)
        assert result == {"a": [1, 2], "b": {"c": 1}}
