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
    with patch("lore.models.requests") as mock_req, \
         patch("lore.models.time.sleep"):
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

def test_start_all_adds_cram_flag_when_host_cache_enabled():
    """start_all() passes -cram <mb> when defaults.host_cache is true."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()), \
         patch("lore.models.ModelServer.health_check", return_value=True):

        mock_popen.return_value = MagicMock(pid=123)

        from lore.models import ModelServer
        config = {
            "primary": {"path": "models/primary.gguf", "port": 19000},
            "defaults": {"host_cache": True, "host_cache_mb": 4096},
        }
        server = ModelServer(config)
        server.start_all()

        args = mock_popen.call_args[0][0]
        assert "-cram" in args
        assert args[args.index("-cram") + 1] == "4096"

def test_start_all_omits_cram_flag_by_default():
    """start_all() does not pass -cram when host_cache is unset/false."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()), \
         patch("lore.models.ModelServer.health_check", return_value=True):

        mock_popen.return_value = MagicMock(pid=123)

        from lore.models import ModelServer
        config = {"primary": {"path": "models/primary.gguf", "port": 19000}}
        server = ModelServer(config)
        server.start_all()

        args = mock_popen.call_args[0][0]
        assert "-cram" not in args


def test_stop_all_closes_log_file_handles():
    """stop_all() closes log file handles to prevent resource leak."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()) as mock_open, \
         patch("lore.models.ModelServer.health_check", return_value=True):

        mock_proc = MagicMock(pid=123)
        mock_popen.return_value = mock_proc
        mock_log = MagicMock()
        mock_open.return_value = mock_log

        from lore.models import ModelServer
        config = {"primary": {"path": "models/primary.gguf", "port": 19000}}
        server = ModelServer(config)
        server.start_all()
        assert "primary" in server._log_files
        server.stop_all()
        mock_log.close.assert_called_once()
        assert server._log_files == {}


def test_start_all_creates_logs_dir():
    """start_all() creates logs/ directory if it doesn't exist."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()), \
         patch("lore.models.Path.mkdir") as mock_mkdir, \
         patch("lore.models.ModelServer.health_check", return_value=True):

        mock_popen.return_value = MagicMock(pid=123)
        from lore.models import ModelServer
        server = ModelServer({"primary": {"path": "models/x.gguf", "port": 19000}})
        server.start_all()
        mock_mkdir.assert_called_once_with(exist_ok=True)


# ─── Public API: is_model_running, start_model, stop_model ───────────────────

def test_is_model_running_true():
    """is_model_running returns True when process exists and poll() is None."""
    from lore.models import ModelServer
    server = ModelServer()
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running
    server._processes["primary"] = mock_proc
    assert server.is_model_running("primary") is True

def test_is_model_running_false_no_process():
    """is_model_running returns False when no process for that role."""
    from lore.models import ModelServer
    server = ModelServer()
    assert server.is_model_running("primary") is False

def test_is_model_running_false_process_dead():
    """is_model_running returns False when process has exited (poll non-None)."""
    from lore.models import ModelServer
    server = ModelServer()
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1  # exited with code 1
    server._processes["primary"] = mock_proc
    assert server.is_model_running("primary") is False

def test_start_model_constructs_correct_args():
    """start_model builds llama-server args from config."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()), \
         patch("lore.models.ModelServer.health_check", return_value=True):
        mock_popen.return_value = MagicMock(pid=42)
        from lore.models import ModelServer
        config = {
            "primary": {"path": "models/primary.gguf", "port": 19000, "context": 16384},
            "defaults": {"context_size": 32768, "kv_cache_type": "turbo4", "flash_attention": True},
        }
        server = ModelServer(config)
        server.start_model("primary")
        args = mock_popen.call_args[0][0]
        assert "-m" in args
        assert "models/primary.gguf" in args
        assert "-c" in args
        assert "16384" in args  # model-specific context, not default
        assert "-ctk" in args and "turbo4" in args
        assert "-fa" in args

def test_start_model_raises_on_missing_file():
    """start_model raises FileNotFoundError when model file doesn't exist."""
    with patch("lore.models.Path.exists", return_value=False):
        from lore.models import ModelServer
        server = ModelServer({"primary": {"path": "missing.gguf", "port": 19000}})
        with pytest.raises(FileNotFoundError):
            server.start_model("primary")

def test_start_model_raises_on_no_config():
    """start_model raises ValueError when no config for role."""
    from lore.models import ModelServer
    server = ModelServer()
    with pytest.raises(ValueError):
        server.start_model("nonexistent")

def test_start_model_raises_on_health_check_failure():
    """start_model raises RuntimeError when health check fails."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()), \
         patch("lore.models.ModelServer.health_check", return_value=False):
        mock_proc = MagicMock(pid=99)
        mock_popen.return_value = mock_proc
        from lore.models import ModelServer
        server = ModelServer({"primary": {"path": "models/x.gguf", "port": 19000}})
        with pytest.raises(RuntimeError):
            server.start_model("primary")
        mock_proc.terminate.assert_called_once()

def test_stop_model_terminates_process():
    """stop_model terminates the process and closes log file."""
    from lore.models import ModelServer
    server = ModelServer()
    mock_proc = MagicMock()
    mock_log = MagicMock()
    server._processes["primary"] = mock_proc
    server._log_files["primary"] = mock_log
    server.stop_model("primary")
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=10)
    mock_log.close.assert_called_once()
    assert "primary" not in server._processes
    assert "primary" not in server._log_files

def test_stop_model_noop_if_not_running():
    """stop_model does nothing if no process for that role."""
    from lore.models import ModelServer
    server = ModelServer()
    server.stop_model("primary")  # should not raise

def test_start_all_delegates_to_start_model():
    """start_all calls start_model for each configured role."""
    with patch("lore.models.ModelServer.start_model") as mock_start, \
         patch("lore.models.Path.exists", return_value=True):
        from lore.models import ModelServer
        config = {
            "primary": {"path": "models/p.gguf", "port": 19000},
            "specialist": {"path": "models/s.gguf", "port": 19001},
        }
        server = ModelServer(config)
        server.start_all()
        roles_called = [call[0][0] for call in mock_start.call_args_list]
        assert "primary" in roles_called
        assert "specialist" in roles_called

def test_stop_all_delegates_to_stop_model():
    """stop_all calls stop_model for each running process."""
    from lore.models import ModelServer
    server = ModelServer()
    server._processes["primary"] = MagicMock()
    server._processes["specialist"] = MagicMock()
    server._log_files["primary"] = MagicMock()
    server._log_files["specialist"] = MagicMock()
    with patch.object(server, "stop_model") as mock_stop:
        server.stop_all()
        roles_called = [call[0][0] for call in mock_stop.call_args_list]
        assert "primary" in roles_called
        assert "specialist" in roles_called


# ─── Server Path Configuration (Issue #8) ──────────────────────────────────

def test_start_model_closes_log_on_popen_failure():
    """start_model closes log file if Popen raises."""
    mock_log = MagicMock()
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", return_value=mock_log):
        mock_popen.side_effect = OSError("exec failed")
        from lore.models import ModelServer
        server = ModelServer({"primary": {"path": "models/x.gguf", "port": 19000}})
        with pytest.raises(OSError):
            server.start_model("primary")
        mock_log.close.assert_called_once()
        assert "primary" not in server._processes
        assert "primary" not in server._log_files


def test_swap_in_closes_log_on_popen_failure():
    """swap_in closes log file if Popen raises."""
    mock_log = MagicMock()
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", return_value=mock_log):
        mock_popen.side_effect = OSError("exec failed")
        from lore.models import ModelServer
        config = {"multimodal": {"path": "models/gemma.gguf", "port": 19003, "context": 16384}}
        server = ModelServer(config)
        with pytest.raises(OSError):
            server.swap_in("gemma")
        mock_log.close.assert_called_once()
        assert "multimodal" not in server._processes


def test_server_path_config_override():
    """engine.server_path in config takes priority."""
    from lore.models import ModelServer
    config = {"engine": {"server_path": "/custom/llama-server"}}
    server = ModelServer(config)
    assert server._cli_path == "/custom/llama-server"

def test_server_path_env_override(monkeypatch):
    """LORE_LLAMA_SERVER env var overrides fallback."""
    monkeypatch.setenv("LORE_LLAMA_SERVER", "/env/llama-server")
    from lore.models import ModelServer
    server = ModelServer()
    assert server._cli_path == "/env/llama-server"

def test_server_path_config_overrides_env(monkeypatch):
    """Config path takes priority over env var."""
    monkeypatch.setenv("LORE_LLAMA_SERVER", "/env/llama-server")
    from lore.models import ModelServer
    config = {"engine": {"server_path": "/config/llama-server"}}
    server = ModelServer(config)
    assert server._cli_path == "/config/llama-server"

def test_server_path_fallback_default(monkeypatch):
    """No config, no env → hardcoded fallback path."""
    monkeypatch.delenv("LORE_LLAMA_SERVER", raising=False)
    from lore.models import ModelServer
    server = ModelServer()
    assert "llama-cpp-turboquant" in server._cli_path
    assert server._cli_path.endswith("llama-server")


# ─── Speculative Decoding (Issue #13) ───────────────────────────────────────

def test_start_model_includes_ngram_simple_for_specialist():
    """start_model adds --spec-type ngram-simple for specialist role."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()), \
         patch("lore.models.ModelServer.health_check", return_value=True):
        mock_popen.return_value = MagicMock(pid=42)
        from lore.models import ModelServer
        config = {
            "specialist": {"path": "models/s.gguf", "port": 19001},
            "defaults": {"speculative_decoding": True},
        }
        server = ModelServer(config)
        server.start_model("specialist")
        args = mock_popen.call_args[0][0]
        assert "--spec-type" in args
        assert "ngram-simple" in args

def test_start_model_no_speculative_for_primary():
    """start_model does NOT add speculative args for primary role."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()), \
         patch("lore.models.ModelServer.health_check", return_value=True):
        mock_popen.return_value = MagicMock(pid=42)
        from lore.models import ModelServer
        config = {
            "primary": {"path": "models/p.gguf", "port": 19000},
            "defaults": {"speculative_decoding": True},
        }
        server = ModelServer(config)
        server.start_model("primary")
        args = mock_popen.call_args[0][0]
        assert "--spec-type" not in args

def test_start_model_speculative_disabled():
    """start_model skips ngram-simple when speculative_decoding is false."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()), \
         patch("lore.models.ModelServer.health_check", return_value=True):
        mock_popen.return_value = MagicMock(pid=42)
        from lore.models import ModelServer
        config = {
            "specialist": {"path": "models/s.gguf", "port": 19001},
            "defaults": {"speculative_decoding": False},
        }
        server = ModelServer(config)
        server.start_model("specialist")
        args = mock_popen.call_args[0][0]
        assert "--spec-type" not in args


# ─── CPU Core Pinning (Issue #14) ───────────────────────────────────────────

def test_pin_cores_calls_taskset_on_linux():
    """_pin_cores calls taskset on Linux with core list."""
    with patch("lore.models.platform.system", return_value="Linux"), \
         patch("lore.models.subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        from lore.models import ModelServer
        config = {"primary": {"cores": [0, 1, 2, 3]}}
        server = ModelServer(config)
        server._pin_cores("primary", 12345)
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert "taskset" in args
        assert "-pc" in args
        assert "0,1,2,3" in args
        assert "12345" in args

def test_pin_cores_skips_when_no_config():
    """_pin_cores does nothing when no cores configured."""
    with patch("lore.models.subprocess.Popen") as mock_popen:
        from lore.models import ModelServer
        server = ModelServer({"primary": {}})
        server._pin_cores("primary", 12345)
        mock_popen.assert_not_called()

def test_pin_cores_handles_failure_gracefully():
    """_pin_cores logs warning but does not raise on subprocess failure."""
    with patch("lore.models.platform.system", return_value="Linux"), \
         patch("lore.models.subprocess.Popen", side_effect=OSError("no such command")):
        from lore.models import ModelServer
        config = {"primary": {"cores": [0, 1]}}
        server = ModelServer(config)
        server._pin_cores("primary", 12345)  # should not raise


# ─── EAGLE-3 Speculative Decoding (Issue #11) ───────────────────────────────

def test_start_model_includes_eagle3_when_configured():
    """start_model adds --spec-type draft-eagle3 when eagle3_draft_path set."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()), \
         patch("lore.models.ModelServer.health_check", return_value=True), \
         patch("lore.models.ModelServer._pin_cores"):
        mock_popen.return_value = MagicMock(pid=42)
        from lore.models import ModelServer
        config = {
            "primary": {"path": "models/p.gguf", "port": 19000,
                        "eagle3_draft_path": "models/eagle3-draft.gguf"},
            "defaults": {},
        }
        server = ModelServer(config)
        server.start_model("primary")
        args = mock_popen.call_args[0][0]
        assert "--spec-type" in args
        assert "draft-eagle3" in args
        assert "-md" in args
        assert "models/eagle3-draft.gguf" in args

def test_start_model_no_eagle3_when_not_configured():
    """start_model does NOT add eagle3 args when eagle3_draft_path not set."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()), \
         patch("lore.models.ModelServer.health_check", return_value=True), \
         patch("lore.models.ModelServer._pin_cores"):
        mock_popen.return_value = MagicMock(pid=42)
        from lore.models import ModelServer
        config = {
            "primary": {"path": "models/p.gguf", "port": 19000},
            "defaults": {},
        }
        server = ModelServer(config)
        server.start_model("primary")
        args = mock_popen.call_args[0][0]
        assert "draft-eagle3" not in args


# ─── Configurable KV Cache Strategy (Track 3) ───────────────────────────────

def test_start_model_uses_configured_kv_cache_type():
    """start_model passes configured kv_cache_type as -ctk/-ctv args."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()), \
         patch("lore.models.ModelServer.health_check", return_value=True), \
         patch("lore.models.ModelServer._pin_cores"):
        mock_popen.return_value = MagicMock(pid=42)
        from lore.models import ModelServer
        config = {
            "primary": {"path": "models/p.gguf", "port": 19000},
            "defaults": {"kv_cache_type": "q8_0"},
        }
        server = ModelServer(config)
        server.start_model("primary")
        args = mock_popen.call_args[0][0]
        idx = args.index("-ctk")
        assert args[idx + 1] == "q8_0"
        idx_v = args.index("-ctv")
        assert args[idx_v + 1] == "q8_0"

def test_start_model_defaults_to_turbo4():
    """start_model defaults to turbo4 when kv_cache_type not specified."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()), \
         patch("lore.models.ModelServer.health_check", return_value=True), \
         patch("lore.models.ModelServer._pin_cores"):
        mock_popen.return_value = MagicMock(pid=42)
        from lore.models import ModelServer
        config = {"primary": {"path": "models/p.gguf", "port": 19000}}
        server = ModelServer(config)
        server.start_model("primary")
        args = mock_popen.call_args[0][0]
        idx = args.index("-ctk")
        assert args[idx + 1] == "turbo4"


# ─── Parallel Slots (-np flag) ──────────────────────────────────────────────

def test_start_model_np_defaults_to_1_without_config():
    """start_model uses -np 1 when parallel_slots not configured."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()), \
         patch("lore.models.ModelServer.health_check", return_value=True), \
         patch("lore.models.ModelServer._pin_cores"):
        mock_popen.return_value = MagicMock(pid=42)
        from lore.models import ModelServer
        config = {"primary": {"path": "models/p.gguf", "port": 19000}}
        server = ModelServer(config)
        server.start_model("primary")
        args = mock_popen.call_args[0][0]
        idx = args.index("-np")
        assert args[idx + 1] == "1"


def test_start_model_np_reads_parallel_slots_from_config():
    """start_model uses -np 3 when parallel_slots=3 in config."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()), \
         patch("lore.models.ModelServer.health_check", return_value=True), \
         patch("lore.models.ModelServer._pin_cores"):
        mock_popen.return_value = MagicMock(pid=42)
        from lore.models import ModelServer
        config = {
            "primary": {"path": "models/p.gguf", "port": 19000, "parallel_slots": 3},
        }
        server = ModelServer(config)
        server.start_model("primary")
        args = mock_popen.call_args[0][0]
        idx = args.index("-np")
        assert args[idx + 1] == "3"


def test_start_model_np_uses_specialist_default():
    """Specialist without parallel_slots config uses -np 1."""
    with patch("lore.models.subprocess.Popen") as mock_popen, \
         patch("lore.models.Path.exists", return_value=True), \
         patch("lore.models.open", MagicMock()), \
         patch("lore.models.ModelServer.health_check", return_value=True), \
         patch("lore.models.ModelServer._pin_cores"):
        mock_popen.return_value = MagicMock(pid=42)
        from lore.models import ModelServer
        config = {"specialist": {"path": "models/s.gguf", "port": 19001}}
        server = ModelServer(config)
        server.start_model("specialist")
        args = mock_popen.call_args[0][0]
        idx = args.index("-np")
        assert args[idx + 1] == "1"


# ─── get_slots() ─────────────────────────────────────────────────────────────

def test_get_slots_returns_slot_data():
    """get_slots() returns slot list from /slots endpoint."""
    with patch("lore.models.requests") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"id": 0, "is_processing": True, "n_past": 150},
            {"id": 1, "is_processing": False, "n_past": 0},
        ]
        mock_req.get.return_value = mock_resp

        from lore.models import ModelServer
        server = ModelServer()
        slots = server.get_slots("primary")

        assert len(slots) == 2
        assert slots[0]["is_processing"] is True
        call_args = mock_req.get.call_args
        assert "/slots" in call_args[0][0]


def test_get_slots_returns_empty_on_error():
    """get_slots() returns empty list on connection error."""
    with patch("lore.models.requests") as mock_req:
        mock_req.get.side_effect = Exception("connection refused")
        mock_req.exceptions.ConnectionError = Exception

        from lore.models import ModelServer
        server = ModelServer()
        slots = server.get_slots("primary")
        assert slots == []


# ─── chat() default timeout is None ─────────────────────────────────────────

def test_chat_default_timeout_is_none():
    """chat() does not set a timeout by default — inference runs to completion."""
    with patch("lore.models.requests") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "hello"}}]}
        mock_req.post.return_value = mock_resp

        from lore.models import ModelServer
        server = ModelServer()
        server.chat("primary", [{"role": "user", "content": "hi"}])

        call_kwargs = mock_req.post.call_args[1]
        assert call_kwargs.get("timeout") is None
