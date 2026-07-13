"""Tests for single-writer orchestration fixes."""
import os
import tempfile
import pytest
from lore.repo_tools import RepoContext


def test_repo_tools_reject_path_traversal():
    """read_file and list_files reject paths escaping repo root."""
    with tempfile.TemporaryDirectory() as tmp:
        # Create a file outside repo
        outside = os.path.join(os.path.dirname(tmp), "secret.txt")
        with open(outside, "w") as f:
            f.write("SECRET")
        try:
            repo = RepoContext(tmp)
            assert "ERROR" in repo.read_file("../secret.txt")
            assert "ERROR" in repo.read_file("../../etc/passwd")
            assert "ERROR" in repo.list_files("..")
        finally:
            os.unlink(outside)
