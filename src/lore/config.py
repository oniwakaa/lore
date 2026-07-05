# src/lore/config.py
"""Central config loader. YAML files + env var overrides."""
import os
from pathlib import Path
import yaml

_ENV_MAP = {
    "LORE_CTX_SIZE": ("models", "defaults", "context_size", int),
    "LORE_PRIMARY_PORT": ("models", "primary", "port", int),
    "LORE_SPECIALIST_PORT": ("models", "specialist", "port", int),
    "LORE_EMBED_PORT": ("models", "embeddings", "port", int),
    "LORE_LOG_LEVEL": ("models", "defaults", "log_level", str),
    "LORE_CONFIDENCE_THRESHOLD": ("router", "confidence_threshold", None, float),
}

class LoreConfig:
    """Loaded configuration from YAML files with env overrides."""

    def __init__(self, config: dict):
        self._config = config

    @property
    def models(self) -> dict:
        return self._config.get("models", {})

    @property
    def router(self) -> dict:
        return self._config.get("router", {})

    @property
    def memory(self) -> dict:
        return self._config.get("memory", {})

    @property
    def session(self) -> dict:
        return self._config.get("session", {})

    @classmethod
    def load(cls, config_dir: str = "configs") -> "LoreConfig":
        """Load all YAML configs from config_dir, apply env overrides."""
        cdir = Path(config_dir)
        config = {}

        for name in ("models", "router", "memory"):
            path = cdir / f"{name}.yaml"
            if path.exists():
                config[name] = yaml.safe_load(path.read_text()) or {}

        # Load sessions config (optional)
        sessions_path = cdir / "sessions.yaml"
        if sessions_path.exists():
            config["session"] = yaml.safe_load(sessions_path.read_text()) or {}

        # Apply env overrides
        for env_key, path_tuple in _ENV_MAP.items():
            val = os.environ.get(env_key)
            if val is None:
                continue
            section, key, subkey, cast = path_tuple
            if section not in config:
                config[section] = {}
            if key not in config[section]:
                config[section][key] = {}
            if subkey is None:
                config[section][key] = cast(val)
            else:
                if not isinstance(config[section][key], dict):
                    config[section][key] = {}
                config[section][key][subkey] = cast(val)

        # Merge context_budget from models into top-level context
        config["context"] = config.get("models", {}).get("context_budget", {})

        return cls(config)
