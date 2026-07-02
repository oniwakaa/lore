# tests/test_integration.py
"""End-to-end integration test. Requires llama-server instances running.
Run with: .venv/bin/pytest tests/test_integration.py -v -s --timeout=120

Start servers first:
  external/llama-cpp-turboquant/build/bin/llama-server \
    -m models/ornith-1.0-9b-Q4_K_M.gguf -c 32768 -ngl 999 -fa on \
    -ctk turbo4 -ctv turbo4 -np 1 --port 19000 --host 127.0.0.1 -fit off &

  external/llama-cpp-turboquant/build/bin/llama-server \
    -m models/Falcon-H1-1.5B-Instruct-Q4_K_M.gguf -c 32768 -ngl 999 -fa on \
    -ctk turbo4 -ctv turbo4 -np 1 --port 19001 --host 127.0.0.1 -fit off &
"""
import pytest

@pytest.mark.integration
def test_full_pipeline_primary():
    """Full pipeline: route -> context -> primary model -> response."""
    from lore.config import LoreConfig
    from lore.models import ModelServer
    from lore.router import Router
    from lore.context import ContextManager

    cfg = LoreConfig.load()
    server = ModelServer(cfg.models)

    if not server.health_check(19000):
        pytest.skip("Primary server not running on port 19000")

    router = Router.load(cfg.router["model_path"])
    ctx = ContextManager(cfg.context, server, system_prompt="You are a helpful assistant.")

    route, confidence = router.classify("Write a Python function to add two numbers")
    assert route == "PRIMARY"

    ctx.add_message("user", "Write a Python function to add two numbers")
    messages = ctx.build_prompt()

    result = server.chat("primary", messages, max_tokens=64, temperature=0)
    content = result["choices"][0]["message"]["content"]
    assert len(content) > 0
    assert "def" in content or "add" in content.lower()

@pytest.mark.integration
def test_full_pipeline_specialist():
    """Full pipeline: route -> specialist model -> response."""
    from lore.config import LoreConfig
    from lore.models import ModelServer
    from lore.router import Router

    cfg = LoreConfig.load()
    server = ModelServer(cfg.models)

    if not server.health_check(19001):
        pytest.skip("Specialist server not running on port 19001")

    router = Router.load(cfg.router["model_path"])
    route, confidence = router.classify("Extract all names from: John went to the store with Mary")
    assert route == "SPECIALIST"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Extract all names from: John went to the store with Mary"},
    ]
    result = server.chat("specialist", messages, max_tokens=64, temperature=0)
    content = result["choices"][0]["message"]["content"]
    assert "John" in content and "Mary" in content
