#!/usr/bin/env python3
"""Start LORE API server with reduced context for local testing."""
import yaml
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from lore.config import LoreConfig
from lore.models import ModelServer
from lore.router import Router
from lore.context import ContextManager
from lore.memory import HierarchicalMemory
from lore.logging import RequestLogger
from lore.tool_attention import ToolAttention
from lore.verifier import Verifier
from lore.api import _app_state, create_server

cfg = LoreConfig.load()
cfg._config["models"]["primary"]["context"] = 16384
cfg._config["models"]["primary"]["parallel_slots"] = 1
cfg._config["models"]["specialist"]["context"] = 16384

server = ModelServer(cfg.models)
print("Starting models...")
server.start_all()
print("Models started.")

router = Router.load(cfg.router.get("model_path", "configs/router_model.joblib"), confidence_threshold=0.40)
sp = "You are a helpful assistant. Answer concisely and accurately."
ts = cfg.models.get("defaults", {}).get("tokenizer_source", "local")
tr = cfg.models.get("primary", {}).get("source", "")
if tr.endswith("-GGUF"): tr = tr[:-len("-GGUF")]
ta = ToolAttention.from_config(server)
cc = yaml.safe_load(Path("configs/compression.yaml").read_text()) if Path("configs/compression.yaml").exists() else {}
mem = HierarchicalMemory(cfg.memory, server)
ctx = ContextManager(cfg.context, server, system_prompt=sp, tokenizer_source=ts,
                     tokenizer_repo=tr or None, tool_attention=ta, compression=cc, memory=mem)
ver = Verifier(yaml.safe_load(Path("configs/verifier.yaml").read_text()) if Path("configs/verifier.yaml").exists() else {})
rl = RequestLogger()
_app_state.update({"cfg": cfg, "server": server, "router": router, "ctx": ctx,
                   "memory": mem, "req_logger": rl, "verifier": ver})
srv = create_server("127.0.0.1", 8000)
print("LORE API on http://127.0.0.1:8000")
srv.serve_forever()
