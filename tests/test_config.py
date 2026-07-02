# tests/test_config.py
import os
import tempfile
import yaml
import pytest
from pathlib import Path

def test_config_loads_all_yaml():
    """Config loader reads models.yaml, router.yaml, memory.yaml."""
    with tempfile.TemporaryDirectory() as d:
        for name, data in [
            ("models.yaml", {"defaults": {"context_size": 32768}, "primary": {"port": 19000}}),
            ("router.yaml", {"confidence_threshold": 0.70}),
            ("memory.yaml", {"top_k": 3}),
        ]:
            Path(d, name).write_text(yaml.dump(data))

        from lore.config import LoreConfig
        cfg = LoreConfig.load(config_dir=d)
        assert cfg.models["defaults"]["context_size"] == 32768
        assert cfg.router["confidence_threshold"] == 0.70
        assert cfg.memory["top_k"] == 3

def test_config_env_overrides():
    """Env vars override YAML values."""
    with tempfile.TemporaryDirectory() as d:
        Path(d, "models.yaml").write_text(yaml.dump(
            {"defaults": {"context_size": 32768}, "primary": {"port": 19000}}))
        Path(d, "router.yaml").write_text(yaml.dump({"confidence_threshold": 0.70}))
        Path(d, "memory.yaml").write_text(yaml.dump({"top_k": 3}))

        os.environ["LORE_CTX_SIZE"] = "8192"
        os.environ["LORE_PRIMARY_PORT"] = "20000"
        os.environ["LORE_CONFIDENCE_THRESHOLD"] = "0.85"
        try:
            from lore.config import LoreConfig
            cfg = LoreConfig.load(config_dir=d)
            assert cfg.models["defaults"]["context_size"] == 8192
            assert cfg.models["primary"]["port"] == 20000
            assert cfg.router["confidence_threshold"] == 0.85
        finally:
            del os.environ["LORE_CTX_SIZE"]
            del os.environ["LORE_PRIMARY_PORT"]
            del os.environ["LORE_CONFIDENCE_THRESHOLD"]
