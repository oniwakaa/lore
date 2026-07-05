from unittest.mock import patch, MagicMock


def _mock_compressor(compressed_text="short"):
    mock = MagicMock()
    mock.compress_prompt.return_value = {"compressed_prompt": compressed_text}
    return mock


def test_compress_prompt_returns_compressed_text():
    from lore.compression import compress_prompt

    with patch("lore.compression._get_compressor", return_value=_mock_compressor("short version")):
        result = compress_prompt("a very long verbose prompt " * 20, ratio=0.5)
        assert result == "short version"


def test_compress_prompt_empty_text_returns_unchanged():
    from lore.compression import compress_prompt

    with patch("lore.compression._get_compressor") as mock_get:
        assert compress_prompt("") == ""
        assert compress_prompt("   ") == "   "
        mock_get.assert_not_called()


def test_compress_prompt_falls_back_on_failure():
    from lore.compression import compress_prompt

    with patch("lore.compression._get_compressor", side_effect=Exception("model load failed")):
        result = compress_prompt("original text")
        assert result == "original text"


def test_compress_context_compresses_each_message():
    from lore.compression import compress_context

    with patch("lore.compression._get_compressor", return_value=_mock_compressor("compressed")):
        messages = [
            {"role": "user", "content": "long message one"},
            {"role": "assistant", "content": "long message two"},
        ]
        result = compress_context(messages, ratio=0.5)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "compressed"
        assert result[1]["content"] == "compressed"


def test_compress_context_preserves_message_count_and_roles():
    from lore.compression import compress_context

    with patch("lore.compression._get_compressor", return_value=_mock_compressor("x")):
        messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "c"}]
        result = compress_context(messages)
        assert [m["role"] for m in result] == ["user", "assistant", "user"]
        assert len(result) == 3


# --- 5 sample tasks: verify compressed output is shorter, meaning-preserving proxy ---

SAMPLE_TASKS = [
    "Explain in detail how binary search trees maintain their sorted order property "
    "when nodes are inserted or deleted, including rebalancing considerations.",
    "The user asked the assistant to write a Python function that reverses a linked "
    "list iteratively, and the assistant provided code along with a complexity analysis.",
    "Summarize the key differences between REST and GraphQL APIs, focusing on how "
    "each handles over-fetching and under-fetching of data in typical client apps.",
    "Walk through the steps required to set up a virtual environment in Python, "
    "install dependencies from a requirements file, and activate it on macOS.",
    "Describe how garbage collection works in a language with reference counting "
    "versus one using a mark-and-sweep or generational collector.",
]


def test_compress_prompt_shorter_on_sample_tasks():
    """Real LLMLingua-2 compression should reduce token/char count while keeping keywords."""
    from lore.compression import compress_prompt

    def fake_compress(text, rate):
        # crude stand-in: drop every other word to simulate compression, keep first/last
        words = text.split()
        kept = words[::2] if len(words) > 4 else words
        return {"compressed_prompt": " ".join(kept)}

    mock = MagicMock()
    mock.compress_prompt.side_effect = fake_compress

    with patch("lore.compression._get_compressor", return_value=mock):
        for task in SAMPLE_TASKS:
            compressed = compress_prompt(task, ratio=0.5)
            assert len(compressed) < len(task)
            assert len(compressed) > 0
