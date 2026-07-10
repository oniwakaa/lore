# tests/test_cli.py
import pytest
from unittest.mock import patch, MagicMock

def test_multimodal_detection_image_path():
    """CLI detects image file paths as multimodal."""
    from lore.cli import is_multimodal
    assert is_multimodal("describe this image: photo.png") is True
    assert is_multimodal("what is in /tmp/cat.jpg") is True
    assert is_multimodal("analyze screenshot.jpeg") is True

def test_multimodal_detection_audio():
    """CLI detects audio file references as multimodal."""
    from lore.cli import is_multimodal
    assert is_multimodal("transcribe recording.wav") is True
    assert is_multimodal("analyze audio.mp3") is True

def test_multimodal_detection_negative():
    """Regular text is not multimodal."""
    from lore.cli import is_multimodal
    assert is_multimodal("write a Python function") is False
    assert is_multimodal("what is 2+2") is False
    assert is_multimodal("extract names from this text") is False

def test_multimodal_detection_image_url():
    """CLI detects image URLs as multimodal."""
    from lore.cli import is_multimodal
    assert is_multimodal("describe https://example.com/image.png") is True
    assert is_multimodal("check this out: http://imgur.com/photo.jpg") is True

def test_cli_single_shot():
    """Single-shot mode routes and returns response."""
    with patch("lore.cli.ModelServer") as mock_ms_class, \
         patch("lore.cli.Router") as mock_router_class, \
         patch("lore.cli.LoreConfig") as mock_cfg_class, \
         patch("lore.cli.TaskClassifier") as mock_classifier_class:

        mock_server = MagicMock()
        mock_server.chat.return_value = {
            "choices": [{"message": {"content": "4"}}]
        }
        mock_ms_class.return_value = mock_server

        mock_router = MagicMock()
        mock_router.classify.return_value = ("PRIMARY", 0.95)
        mock_router_class.load.return_value = mock_router

        mock_cfg = MagicMock()
        mock_cfg.models = {"defaults": {"context_size": 32768}}
        mock_cfg.router = {"confidence_threshold": 0.70, "model_path": "x"}
        mock_cfg.memory = {"top_k": 3, "max_entries": 200, "similarity_threshold": 0.5, "max_text_chars": 500}
        mock_cfg.context = {"working_context": 4096}
        mock_cfg_class.load.return_value = mock_cfg

        from lore.cli import main
        import sys
        with patch.object(sys, "argv", ["lore", "what is 2+2?"]):
            main()

        # Should have called chat on primary (classifier is mocked, so only 1 chat call)
        mock_server.chat.assert_called_once()
        call_args = mock_server.chat.call_args
        assert call_args[0][0] == "primary"


def test_process_single_and_repl_share_dispatch():
    """_process_single and _run_repl both route through the shared _dispatch()."""
    from lore import cli
    import inspect

    assert "_dispatch(" in inspect.getsource(cli._process_single)
    assert "_dispatch(" in inspect.getsource(cli._run_repl)


def test_dispatch_tool_only_skips_chat():
    """_dispatch resolves TOOL_ONLY queries via tool_handler without calling chat()."""
    from lore.cli import _dispatch

    mock_server = MagicMock()
    mock_router = MagicMock()
    mock_router.classify.return_value = ("TOOL_ONLY", 0.9)
    mock_ctx = MagicMock()
    mock_ctx.was_truncated = False
    mock_memory = MagicMock()
    mock_memory.retrieve.return_value = []
    mock_logger = MagicMock()

    result = _dispatch("2+2", mock_server, mock_router, mock_ctx, mock_memory, mock_logger, False)

    assert result["content"] == "4"
    assert result["model"] == "tool_handler"
    mock_server.chat.assert_not_called()


def test_process_single_non_multimodal_error_surfaces_real_message(capsys):
    """_process_single surfaces real error message, not 'multimodal unavailable'."""
    from lore.cli import _process_single

    mock_server = MagicMock()
    mock_router = MagicMock()
    mock_ctx = MagicMock()
    mock_memory = MagicMock()
    mock_logger = MagicMock()

    # Simulate a non-multimodal error (e.g., router failure)
    mock_router.classify.side_effect = RuntimeError("router model not loaded")

    _process_single("test query", mock_server, mock_router, mock_ctx,
                    mock_memory, mock_logger, False)

    captured = capsys.readouterr()
    assert "multimodal" not in captured.err.lower()
    assert "router model not loaded" in captured.err
    mock_server.stop_all.assert_called_once()


# ─── _resolve_route tests (Issue #18) ────────────────────────────────────────

def test_resolve_route_primary():
    """_resolve_route returns PRIMARY route with correct model."""
    from lore.cli import _resolve_route
    mock_router = MagicMock()
    mock_router.classify.return_value = ("PRIMARY", 0.95)
    route, confidence, model = _resolve_route("write a function", mock_router)
    assert route == "PRIMARY"
    assert confidence == 0.95
    assert model == "primary"


def test_resolve_route_specialist():
    """_resolve_route returns SPECIALIST route with specialist model."""
    from lore.cli import _resolve_route
    mock_router = MagicMock()
    mock_router.classify.return_value = ("SPECIALIST", 0.88)
    route, confidence, model = _resolve_route("extract names", mock_router)
    assert route == "SPECIALIST"
    assert model == "specialist"


def test_resolve_route_multimodal():
    """_resolve_route returns MULTIMODAL for image references."""
    from lore.cli import _resolve_route
    mock_router = MagicMock()
    route, confidence, model = _resolve_route("describe photo.png", mock_router)
    assert route == "MULTIMODAL"
    assert confidence == 1.0
    assert model == "multimodal"
    mock_router.classify.assert_not_called()


# ─── _execute_query tests (Issue #18) ────────────────────────────────────────

def test_execute_query_success():
    """_execute_query returns content and success on normal chat."""
    from lore.cli import _execute_query
    mock_server = MagicMock()
    mock_server.chat.return_value = {"choices": [{"message": {"content": "result"}}]}
    mock_ctx = MagicMock()
    mock_memory = MagicMock()
    mock_memory.retrieve.return_value = []

    content, success = _execute_query("query", "primary", mock_server, mock_ctx, mock_memory, False)
    assert content == "result"
    assert success is True


def test_execute_query_specialist_fallback():
    """_execute_query falls back to primary on specialist failure."""
    from lore.cli import _execute_query
    mock_server = MagicMock()
    mock_server.chat.side_effect = [
        Exception("specialist error"),
        {"choices": [{"message": {"content": "primary result"}}]},
    ]
    mock_ctx = MagicMock()
    mock_memory = MagicMock()
    mock_memory.retrieve.return_value = []

    content, success = _execute_query("query", "specialist", mock_server, mock_ctx, mock_memory, False)
    assert content == "primary result"
    assert success is True
    assert mock_server.chat.call_count == 2


def test_execute_query_primary_failure():
    """_execute_query returns error on primary failure."""
    from lore.cli import _execute_query
    mock_server = MagicMock()
    mock_server.chat.side_effect = Exception("server down")
    mock_ctx = MagicMock()
    mock_memory = MagicMock()
    mock_memory.retrieve.return_value = []

    content, success = _execute_query("query", "primary", mock_server, mock_ctx, mock_memory, False)
    assert not success
    assert "Error" in content
