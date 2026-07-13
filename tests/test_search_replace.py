# tests/test_search_replace.py
"""Tests for SEARCH/REPLACE block parser and fuzzy applicator."""
from lore.search_replace import (
    parse_edit_blocks,
    apply_edit_blocks,
    _apply_single_edit,
    _normalize_whitespace,
    _strip_edge_blank_lines,
    _to_relative_indent,
)


class TestParseEditBlocks:
    """Test SEARCH/REPLACE block parsing."""

    def test_basic_block(self):
        content = """path/to/file.py
<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE"""
        blocks = parse_edit_blocks(content)
        assert len(blocks) == 1
        assert blocks[0] == ("path/to/file.py", "old code", "new code")

    def test_multiple_blocks(self):
        content = """file1.py
<<<<<<< SEARCH
code1
=======
new1
>>>>>>> REPLACE

file2.py
<<<<<<< SEARCH
code2
=======
new2
>>>>>>> REPLACE"""
        blocks = parse_edit_blocks(content)
        assert len(blocks) == 2
        assert blocks[0][0] == "file1.py"
        assert blocks[1][0] == "file2.py"

    def test_multiline_search_replace(self):
        content = """django/models.py
<<<<<<< SEARCH
def foo():
    pass
=======
def foo():
    return 42
>>>>>>> REPLACE"""
        blocks = parse_edit_blocks(content)
        assert len(blocks) == 1
        assert "def foo():" in blocks[0][1]
        assert "return 42" in blocks[0][2]

    def test_no_blocks(self):
        content = "Just some regular text with no edit blocks."
        blocks = parse_edit_blocks(content)
        assert len(blocks) == 0

    def test_filepath_from_backticks(self):
        content = """```python
file.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
```"""
        blocks = parse_edit_blocks(content)
        assert len(blocks) == 1
        assert blocks[0][0] == "file.py"

    def test_flexible_markers(self):
        """Aider accepts 5-9 marker chars."""
        content = """file.py
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE"""
        blocks = parse_edit_blocks(content)
        assert len(blocks) == 1


class TestApplySingleEdit:
    """Test single SEARCH/REPLACE application with fuzzy matching."""

    def test_exact_match(self):
        content = "line1\nline2\nline3\n"
        result = _apply_single_edit(content, "line2", "replaced")
        assert result == "line1\nreplaced\nline3\n"

    def test_no_match(self):
        content = "line1\nline2\nline3\n"
        result = _apply_single_edit(content, "nonexistent", "replaced")
        assert result is None

    def test_whitespace_normalized(self):
        content = "line1\n  line2\nline3\n"
        result = _apply_single_edit(content, "line2", "replaced")
        # Should match via whitespace normalization
        assert result is not None
        assert "replaced" in result

    def test_blank_line_stripping(self):
        content = "line1\n\nline2\nline3\n"
        result = _apply_single_edit(content, "\nline2", "replaced")
        assert result is not None

    def test_multiline_replace(self):
        content = "def foo():\n    pass\n"
        result = _apply_single_edit(content, "def foo():\n    pass", "def foo():\n    return 1")
        assert result is not None
        assert "return 1" in result

    def test_first_match_only(self):
        content = "a\nb\na\nb\n"
        result = _apply_single_edit(content, "a\nb", "x\ny")
        assert result is not None
        # Should replace first occurrence only
        assert result.count("x\ny") == 1

    def test_empty_search_appends(self):
        content = "existing"
        result = _apply_single_edit(content, "", "new")
        assert result == "existingnew"


class TestApplyEditBlocks:
    """Test applying multiple edit blocks."""

    def test_single_block(self):
        content = "line1\nline2\nline3\n"
        blocks = [("file.py", "line2", "replaced")]
        result = apply_edit_blocks(content, blocks)
        assert result == "line1\nreplaced\nline3\n"

    def test_multiple_blocks_sequential(self):
        content = "a\nb\nc\n"
        blocks = [
            ("file.py", "a", "x"),
            ("file.py", "b", "y"),
        ]
        result = apply_edit_blocks(content, blocks)
        assert result is not None
        assert "x" in result
        assert "y" in result

    def test_failed_block_returns_none(self):
        content = "a\nb\n"
        blocks = [
            ("file.py", "a", "x"),  # succeeds
            ("file.py", "nonexistent", "y"),  # fails
        ]
        result = apply_edit_blocks(content, blocks)
        assert result is None


class TestHelpers:
    """Test utility functions."""

    def test_strip_edge_blank_lines(self):
        assert _strip_edge_blank_lines("\n\nhello\n\n") == "hello"

    def test_normalize_whitespace(self):
        assert _normalize_whitespace("  hello\tworld  ") == "  hello    world"

    def test_relative_indent(self):
        result = _to_relative_indent("    foo\n        bar\n    baz")
        # First line: 4 spaces
        # Second line: 8 - 4 = 4 more spaces
        # Third line: 4 - 4 = 0 more spaces
        assert "foo" in result
        assert "bar" in result
        assert "baz" in result
