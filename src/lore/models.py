# src/lore/models.py
"""Model server lifecycle + HTTP client. Manages llama-server instances."""
import subprocess
import time
import logging
import os
import platform
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
        self._cli_path = self._resolve_server_path()

    def _resolve_server_path(self) -> str:
        """Resolve llama-server binary path: config > env var > bundled fallback."""
        # 1. engine.server_path in config
        engine_cfg = self._config.get("engine", {})
        configured = engine_cfg.get("server_path")
        if configured:
            return configured
        # 2. LORE_LLAMA_SERVER env var
        env_path = os.environ.get("LORE_LLAMA_SERVER")
        if env_path:
            return env_path
        # 3. Hardcoded fallback
        return str(Path(__file__).parent.parent.parent /
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

        parallel_slots = mcfg.get("parallel_slots", 1)

        args = [
            self._cli_path, "-m", path,
            "-c", str(mctx), "-ngl", str(ngl),
            "-np", str(parallel_slots), "--port", str(port),
            "--host", "127.0.0.1",
            "-fit", "off",  # ponytail: -fit on hangs on M4 with 9B model; off = instant load
        ]
        if is_embed:
            args.append("--embedding")
        else:
            args += ["-fa", fa, "-ctk", kv, "-ctv", kv]
        if host_cache:
            args += ["-cram", str(host_cache_mb)]

        # Speculative decoding: ngram-simple for specialist (no draft model needed)
        if role == "specialist" and defaults.get("speculative_decoding", True):
            args += ["--spec-type", "ngram-simple"]

        # EAGLE-3 speculative decoding for primary (requires draft model path)
        if role == "primary" and mcfg.get("eagle3_draft_path"):
            args += ["--spec-type", "draft-eagle3", "-md", mcfg["eagle3_draft_path"]]

        # MTP speculative decoding for primary (built-in MTP head, no draft model)
        if role == "primary" and mcfg.get("spec_type"):
            args += ["--spec-type", mcfg["spec_type"]]
            if mcfg.get("spec_draft_n_max"):
                args += ["--spec-draft-n-max", str(mcfg["spec_draft_n_max"])]

        log_file = open(f"logs/{role}.log", "w")
        try:
            proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=log_file)
        except Exception:
            log_file.close()
            raise
        self._processes[role] = proc
        self._log_files[role] = log_file
        logger.info(f"Started {role} on port {port} (PID {proc.pid})")

        if not self.health_check(port):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            del self._processes[role]
            fh = self._log_files.pop(role, None)
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass
            raise RuntimeError(f"Model {role} failed health check")

        self._pin_cores(role, proc.pid)

    def _pin_cores(self, role: str, pid: int) -> None:
        """Pin model process to specific CPU cores if configured."""
        cores_cfg = self._config.get(role, {}).get("cores")
        if not cores_cfg:
            return
        try:
            if platform.system() == "Darwin":
                subprocess.Popen(["taskpolicy", "-b", "-p", str(pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                core_list = ",".join(str(c) for c in cores_cfg)
                subprocess.Popen(["taskset", "-pc", core_list, str(pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"Pinned {role} (PID {pid}) to cores {cores_cfg}")
        except Exception as e:
            logger.warning(f"Core pinning failed for {role}: {e}")

    def stop_model(self, role: str) -> None:
        """Stop a single model server by role name."""
        proc = self._processes.pop(role, None)
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
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

    def health_check(self, port: int, retries: int = 60) -> bool:
        """Check if server on port responds to /health.

        Default 60 retries (1s apart) = 60s window. 9B model cold start
        from SSD can take 10-30s; 3 retries was too aggressive.
        Server returns 503 while loading — must sleep on non-200 too,
        not just on connection errors.
        """
        for _ in range(retries):
            try:
                resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def chat(self, model: str, messages: list[dict], **opts) -> dict:
        """POST to /v1/chat/completions. opts: max_tokens, temperature, response_format.

        No timeout by default — inference takes as long as it takes.
        Callers can still pass timeout=N for non-inference calls if needed.

        If the model role has a 'sampling' config in models.yaml, those params
        override caller opts. This lets reasoning models force their required
        sampling (e.g., Qwythos needs temperature=0.6 to avoid repetition loops).
        """
        # Extract timeout before building body (don't leak into JSON)
        timeout = opts.pop("timeout", None)
        # Apply model-specific sampling overrides from config
        model_cfg = self._config.get(model, {})
        sampling = model_cfg.get("sampling")
        if sampling:
            for k, v in sampling.items():
                opts[k] = v
        body = {"messages": messages, "stream": False}
        body.update(opts)
        resp = requests.post(self._url(model, "/v1/chat/completions"), json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def get_slots(self, model: str) -> list[dict]:
        """GET /slots — returns slot status from llama-server.

        Each slot dict has: id, is_processing, n_ctx, prompt, n_past, etc.
        Used by orchestrator for intelligent supervision (checking if
        a subtask is still generating tokens).
        """
        try:
            resp = requests.get(self._url(model, "/slots"), timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug(f"get_slots failed for {model}: {e}")
            return []

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
        # Re-entrancy guard: stop existing multimodal if already running
        if self.is_model_running("multimodal"):
            self.stop_model("multimodal")
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
            "-fit", "off",
        ]
        log_file = open("logs/multimodal.log", "w")
        try:
            proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=log_file)
        except Exception:
            log_file.close()
            raise
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
