# src/lore/models.py
"""Model server lifecycle + HTTP client. Manages llama-server instances."""
import subprocess
import time
import logging
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

class ModelServer:
    """Manages llama-server instances and dispatches HTTP requests."""

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._log_files: dict[str, object] = {}  # file handles for cleanup
        self._ports = {
            "primary": self._config.get("primary", {}).get("port", 19000),
            "specialist": self._config.get("specialist", {}).get("port", 19001),
            "embeddings": self._config.get("embeddings", {}).get("port", 19002),
            "multimodal": self._config.get("multimodal", {}).get("port", 19003),
        }
        self._cli_path = str(Path(__file__).parent.parent.parent /
                             "external/llama-cpp-turboquant/build/bin/llama-server")

    def _port_for(self, model: str) -> int:
        return self._ports.get(model, 19000)

    def _url(self, model: str, path: str) -> str:
        return f"http://127.0.0.1:{self._port_for(model)}{path}"

    def is_model_running(self, role: str) -> bool:
        """Check if a model server process is running for the given role."""
        proc = self._processes.get(role)
        return proc is not None and proc.poll() is None

    def start_model(self, role: str) -> None:
        """Start a single model server by role name (e.g., 'specialist').

        Uses the same args construction as start_all() but for one model.
        Raises FileNotFoundError if model file is missing.
        Raises RuntimeError if health check fails.
        """
        Path("logs").mkdir(exist_ok=True)
        mcfg = self._config.get(role)
        if not mcfg:
            raise ValueError(f"No config for model role '{role}'")
        path = mcfg.get("path")
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"Model file missing for {role}: {path}")

        defaults = self._config.get("defaults", {})
        port = mcfg.get("port", self._ports.get(role, 19000))
        mctx = mcfg.get("context", defaults.get("context_size", 32768))
        kv = defaults.get("kv_cache_type", "turbo4")
        fa = "on" if defaults.get("flash_attention", True) else "off"
        ngl = defaults.get("gpu_layers", 999)
        host_cache = defaults.get("host_cache", False)
        host_cache_mb = defaults.get("host_cache_mb", 8192)
        is_embed = role == "embeddings"

        args = [
            self._cli_path, "-m", path,
            "-c", str(mctx), "-ngl", str(ngl),
            "-np", "1", "--port", str(port),
            "--host", "127.0.0.1",
        ]
        if is_embed:
            args.append("--embedding")
        else:
            args += ["-fa", fa, "-ctk", kv, "-ctv", kv]
        if host_cache:
            args += ["-cram", str(host_cache_mb)]

        log_file = open(f"logs/{role}.log", "w")
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=log_file)
        self._processes[role] = proc
        self._log_files[role] = log_file
        logger.info(f"Started {role} on port {port} (PID {proc.pid})")

        if not self.health_check(port):
            proc.terminate()
            raise RuntimeError(f"Model {role} failed health check")

    def stop_model(self, role: str) -> None:
        """Stop a single model server by role name."""
        proc = self._processes.pop(role, None)
        if proc:
            proc.terminate()
            proc.wait(timeout=10)
            logger.info(f"Stopped {role}")
        fh = self._log_files.pop(role, None)
        if fh:
            try:
                fh.close()
            except Exception:
                pass

    def start_all(self) -> None:
        """Start all persistent llama-server instances."""
        for role in ("primary", "specialist", "embeddings"):
            mcfg = self._config.get(role)
            if not mcfg:
                continue
            try:
                self.start_model(role)
            except FileNotFoundError as e:
                logger.warning(f"Model file missing for {role}: {e}")
            except RuntimeError as e:
                logger.error(f"Health check failed for {role}: {e}")
            except Exception as e:
                logger.error(f"Failed to start {role}: {e}")

    def stop_all(self) -> None:
        """Graceful shutdown of all servers."""
        for role in list(self._processes.keys()):
            self.stop_model(role)

    def health_check(self, port: int) -> bool:
        """Check if server on port responds to /health."""
        for _ in range(3):
            try:
                resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
                if resp.status_code == 200:
                    return True
            except Exception:
                time.sleep(1)
        return False

    def chat(self, model: str, messages: list[dict], **opts) -> dict:
        """POST to /v1/chat/completions. opts: max_tokens, temperature, response_format."""
        body = {"messages": messages, "stream": False}
        body.update(opts)
        resp = requests.post(self._url(model, "/v1/chat/completions"), json=body, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def tokenize(self, model: str, text: str) -> int:
        """POST to /tokenize, return token count."""
        resp = requests.post(self._url(model, "/tokenize"), json={"content": text}, timeout=10)
        resp.raise_for_status()
        return len(resp.json().get("tokens", []))

    def embed(self, text: str) -> list[float]:
        """POST to /v1/embeddings on embeddings server."""
        resp = requests.post(
            self._url("embeddings", "/v1/embeddings"),
            json={"input": text, "model": "nomic-embed"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def swap_in(self, model_name: str) -> None:
        """Start a swap model (e.g., Gemma 4 E4B) as a new process."""
        Path("logs").mkdir(exist_ok=True)
        mcfg = self._config.get("multimodal", {})
        path = mcfg.get("path")
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"Swap model not found: {path}")
        port = mcfg.get("port", 19003)
        ctx = mcfg.get("context", 16384)
        args = [
            self._cli_path, "-m", path,
            "-c", str(ctx), "-ngl", "999", "-fa", "on",
            "-ctk", "turbo4", "-ctv", "turbo4",
            "-np", "1", "--port", str(port),
            "--host", "127.0.0.1",
        ]
        log_file = open("logs/multimodal.log", "w")
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=log_file)
        self._processes["multimodal"] = proc
        self._log_files["multimodal"] = log_file
        # Health check
        if not self.health_check(port):
            proc.terminate()
            raise RuntimeError(f"Swap model {model_name} failed health check")

    def swap_out(self, model_name: str) -> None:
        """Kill a swap model process."""
        self.stop_model("multimodal")

    def verify_prefix_cache(self) -> bool:
        """Send identical prompt twice, check if second is faster (cache hit)."""
        try:
            msgs = [{"role": "user", "content": "cache test ping"}]
            t0 = time.time()
            self.chat("primary", msgs, max_tokens=1, temperature=0)
            t1 = time.time()
            self.chat("primary", msgs, max_tokens=1, temperature=0)
            t2 = time.time()
            cached = (t2 - t1) < (t1 - t0) * 0.7
            logger.info(f"Prefix cache: first={t1-t0:.2f}s, second={t2-t1:.2f}s, cached={cached}")
            return cached
        except Exception as e:
            logger.warning(f"Prefix cache verification failed: {e}")
            return False
