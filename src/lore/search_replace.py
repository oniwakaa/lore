"""SEARCH/REPLACE block parser and fuzzy applicator.

Implements Aider-style edit blocks with multi-strategy fuzzy matching.
This replaces unified diffs as LORE's code editing format because 9B Q4
models produce correct code but hallucinate line numbers and context lines.

Format:
    path/to/file.py
    <<<<<<< SEARCH
    original code to find
    =======
    replacement code
    >>>>>>> REPLACE

Matching cascade (inspired by Aider's replace_most_similar_chunk):
1. Exact match
2. Strip leading/trailing blank lines
3. Normalize whitespace (tabs→spaces, trailing whitespace)
4. Relative indentation matching
5. Substring containment (SEARCH is a subset of a larger block)
6. SequenceMatcher fuzzy (threshold 0.6)

If all strategies fail, returns None (edit not applied).
"""
import re
import difflib
from pathlib import Path

# Parse patterns — flexible on the number of markers (5-9 chars)
_HEAD = re.compile(r"^<{5,9}\s*SEARCH>?\s*$")
_DIVIDER = re.compile(r"^={5,9}\s*$")
_TAIL = re.compile(r"^>{5,9}\s*REPLACE\s*$")


def parse_edit_blocks(content: str) -> list[tuple[str, str, str]]:
    """Extract SEARCH/REPLACE blocks from model output.

    Returns list of (filepath, search_text, replace_text).
    """
    blocks = []
    lines = content.split("\n")
    i = 0

    while i < len(lines):
        # Look for SEARCH marker
        if _HEAD.match(lines[i].strip()):
            # Walk backwards to find the filepath
            filepath = _find_filename(lines, i)
            if not filepath:
                i += 1
                continue

            # Collect SEARCH lines
            i += 1
            search_lines = []
            while i < len(lines) and not _DIVIDER.match(lines[i].strip()):
                search_lines.append(lines[i])
                i += 1

            if i >= len(lines):
                break  # malformed — no divider

            # Skip divider
            i += 1

            # Collect REPLACE lines
            replace_lines = []
            while i < len(lines) and not _TAIL.match(lines[i].strip()):
                replace_lines.append(lines[i])
                i += 1

            search_text = "\n".join(search_lines)
            replace_text = "\n".join(replace_lines)
            blocks.append((filepath, search_text, replace_text))

        i += 1

    return blocks


def _find_filename(lines: list[str], search_idx: int) -> str | None:
    """Walk backwards from SEARCH marker to find the filepath.

    Looks up to 5 lines back for a line that looks like a file path.
    """
    for j in range(search_idx - 1, max(search_idx - 6, -1), -1):
        line = lines[j].strip()
        # Strip markdown fences and backticks
        line = line.strip("`").strip()
        if not line:
            continue
        # Looks like a file path?
        if "/" in line or line.endswith(".py") or line.endswith(".js") or line.endswith(".ts"):
            # Strip leading/trailing markdown
            line = line.strip("*").strip("`").strip()
            return line
        # Could be a bare filename
        if "." in line and " " not in line and len(line) < 100:
            return line
    return None


def apply_edit_blocks(content: str, blocks: list[tuple[str, str, str]]) -> str | None:
    """Apply SEARCH/REPLACE blocks to file content.

    Tries multiple matching strategies for each block.
    Returns modified content, or None if any block fails to match.
    """
    result = content
    for filepath, search_text, replace_text in blocks:
        new_result = _apply_single_edit(result, search_text, replace_text)
        if new_result is None:
            return None  # block failed — caller should report which one
        result = new_result
    return result


def _apply_single_edit(content: str, search: str, replace: str) -> str | None:
    """Apply a single SEARCH/REPLACE edit with fuzzy matching cascade."""
    if not search.strip():
        # Empty SEARCH = append (new file or append to existing)
        return content + replace

    # Strategy 1: Exact match
    if search in content:
        return content.replace(search, replace, 1)

    # Strategy 2: Strip leading/trailing blank lines from both
    s_stripped = _strip_edge_blank_lines(search)
    if s_stripped and s_stripped in content:
        r_stripped = _strip_edge_blank_lines(replace)
        return content.replace(s_stripped, r_stripped, 1)

    # Strategy 3: Normalize whitespace (tabs→spaces, trailing whitespace)
    s_norm_lines = _normalize_whitespace(search).split("\n")
    c_norm_lines = _normalize_whitespace(content).split("\n")
    match = _find_line_range(c_norm_lines, s_norm_lines)
    if match is not None:
        start, end = match
        c_lines = content.split("\n")
        r_lines = replace.split("\n")
        return "\n".join(c_lines[:start] + r_lines + c_lines[end:])

    # Strategy 4: Relative indentation matching
    s_rel_lines = _to_relative_indent(search).split("\n")
    c_rel_lines = _to_relative_indent(content).split("\n")
    match = _find_line_range(c_rel_lines, s_rel_lines)
    if match is not None:
        start, end = match
        c_lines = content.split("\n")
        r_lines = replace.split("\n")
        return "\n".join(c_lines[:start] + r_lines + c_lines[end:])

    # Strategy 5: Substring containment — SEARCH is a subset of a larger block
    s_lines = search.strip().split("\n")
    c_lines = content.split("\n")
    match = _find_containing_lines(c_lines, s_lines)
    if match is not None:
        start, end = match
        new_lines = c_lines[:start] + replace.split("\n") + c_lines[end:]
        return "\n".join(new_lines)

    # Strategy 6: SequenceMatcher fuzzy (threshold 0.6)
    best_match = _fuzzy_find(c_lines, s_lines, threshold=0.6)
    if best_match is not None:
        start, end = best_match
        new_lines = c_lines[:start] + replace.split("\n") + c_lines[end:]
        return "\n".join(new_lines)

    return None  # all strategies failed


def _strip_edge_blank_lines(text: str) -> str:
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return "\n".join(lines)


def _normalize_whitespace(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(line.rstrip().replace("\t", "    ") for line in lines)


def _to_relative_indent(text: str) -> str:
    """Convert indentation to relative (difference from previous line)."""
    lines = text.split("\n")
    result = []
    prev_indent = 0
    for line in lines:
        indent = len(line) - len(line.lstrip())
        diff = indent - prev_indent
        if diff >= 0:
            result.append(" " * diff + line.lstrip())
        else:
            result.append(line.lstrip())  # outdent — just strip
        prev_indent = indent
    return "\n".join(result)


def _find_line_range(content_lines: list[str], search_lines: list[str]) -> tuple[int, int] | None:
    """Find search_lines as a contiguous sublist of content_lines (exact match)."""
    if not search_lines:
        return None
    n, m = len(content_lines), len(search_lines)
    for i in range(n - m + 1):
        if content_lines[i:i + m] == search_lines:
            return (i, i + m)
    return None


def _find_containing_lines(content_lines: list[str], search_lines: list[str]) -> tuple[int, int] | None:
    """Find if search_lines appear as a contiguous subset of content_lines (whitespace-insensitive)."""
    if not search_lines:
        return None
    target = [l.strip() for l in search_lines]
    for i, line in enumerate(content_lines):
        if line.strip() == target[0]:
            end = i + len(search_lines)
            if end > len(content_lines):
                continue
            candidate = [l.strip() for l in content_lines[i:end]]
            if candidate == target:
                return (i, end)
    return None


def _fuzzy_find(content_lines: list[str], search_lines: list[str],
                threshold: float = 0.6) -> tuple[int, int] | None:
    """Find the best fuzzy match of search_lines in content_lines."""
    if not search_lines:
        return None

    best_ratio = 0
    best_start = 0
    best_end = 0

    search_text = "\n".join(l.strip() for l in search_lines)

    for length in range(max(1, len(search_lines) - 2), len(search_lines) + 3):
        for i in range(len(content_lines) - length + 1):
            chunk = "\n".join(l.strip() for l in content_lines[i:i + length])
            ratio = difflib.SequenceMatcher(None, search_text, chunk).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
                best_end = i + length

    if best_ratio >= threshold:
        return (best_start, best_end)
    return None
