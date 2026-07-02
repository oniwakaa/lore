# tests/test_models.py
import pytest
from unittest.mock import patch, MagicMock
import json

def test_chat_calls_correct_endpoint():
    """chat() POSTs to /v1/chat/completions on the right port."""
    with patch("lore.models.requests") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "hello"}}]}
        mock_req.post.return_value = mock_resp

        from lore.models import ModelServer
        server = ModelServer()
        result = server.chat("primary", [{"role": "user", "content": "hi"}])

        assert result["choices"][0]["message"]["content"] == "hello"
        call_args = mock_req.post.call_args
        assert "/v1/chat/completions" in call_args[0][0]

def test_tokenize_returns_int():
    """tokenize() returns token count as int."""
    with patch("lore.models.requests") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"tokens": ["t1", "t2", "t3"]}
        mock_req.post.return_value = mock_resp

        from lore.models import ModelServer
        server = ModelServer()
        count = server.tokenize("primary", "hello world")
        assert count == 3

def test_embed_calls_v1_embeddings():
    """embed() POSTs to /v1/embeddings endpoint."""
    with patch("lore.models.requests") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_req.post.return_value = mock_resp

        from lore.models import ModelServer
        server = ModelServer()
        embedding = server.embed("hello")
        assert len(embedding) == 3
        call_args = mock_req.post.call_args
        assert "/v1/embeddings" in call_args[0][0]

def test_health_check_returns_false_on_connection_error():
    """health_check returns False when server is not running."""
    with patch("lore.models.requests") as mock_req:
        mock_req.exceptions.ConnectionError = Exception
        mock_req.get.side_effect = Exception("connection refused")

        from lore.models import ModelServer
        server = ModelServer()
        assert server.health_check(99999) is False

def test_chat_with_json_response_format():
    """chat() passes response_format for GBNF constrained decoding."""
    with patch("lore.models.requests") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        mock_req.post.return_value = mock_resp

        from lore.models import ModelServer
        server = ModelServer()
        server.chat("primary", [{"role": "user", "content": "test"}],
                    response_format={"type": "json_object"})

        call_body = mock_req.post.call_args[1]["json"]
        assert call_body.get("response_format") == {"type": "json_object"}
