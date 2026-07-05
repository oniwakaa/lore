from lore.tool_handler import handle_tool_only


def test_simple_math():
    assert handle_tool_only("2+2") == "4"
    assert handle_tool_only("10 * 5") == "50"
    assert handle_tool_only("7 / 2") == "3.5"
    assert handle_tool_only("10 - 3") == "7"


def test_date_query():
    from datetime import date
    assert handle_tool_only("what is today's date") == date.today().isoformat()
    assert handle_tool_only("current date") == date.today().isoformat()


def test_unit_conversion_distance():
    assert handle_tool_only("10 km to miles") == "6.2137 miles"


def test_unit_conversion_temperature():
    assert handle_tool_only("100 c to f") == "212 f"
    assert handle_tool_only("32 f to c") == "0 c"


def test_json_format_detection_valid():
    assert handle_tool_only('is this valid json: {"a": 1}') == "Valid JSON"


def test_json_format_detection_invalid():
    result = handle_tool_only("is this valid json: {a: 1}")
    assert result.startswith("Invalid JSON")


def test_no_match_returns_none():
    assert handle_tool_only("write a Python function to sort a list") is None
    assert handle_tool_only("summarize this article for me") is None
