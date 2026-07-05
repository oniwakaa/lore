"""LLMLingua-2 prompt compression. Reduces noise in old context, not just tokens."""
import logging

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

_compressor = None  # lazy singleton; model load takes seconds + downloads on first use


def _get_compressor(model_path: str, device_map: str):
    global _compressor
    if _compressor is None:
        from llmlingua import PromptCompressor
        _compressor = PromptCompressor(model_name=model_path, use_llmlingua2=True, device_map=device_map)
    return _compressor


def compress_prompt(text: str, ratio: float = 0.5, model_path: str = _DEFAULT_MODEL,
                     device_map: str = "cpu") -> str:
    """Compress text via LLMLingua-2. Returns original text unchanged on failure."""
    if not text or not text.strip():
        return text
    try:
        compressor = _get_compressor(model_path, device_map)
        result = compressor.compress_prompt(text, rate=ratio)
        return result["compressed_prompt"]
    except Exception as e:
        logger.warning(f"Prompt compression failed ({e}), returning original text")
        return text


def compress_context(messages: list[dict], ratio: float = 0.5, model_path: str = _DEFAULT_MODEL,
                      device_map: str = "cpu") -> list[dict]:
    """Compress each message's content in a list. Roles are unaffected."""
    return [
        {**m, "content": compress_prompt(m["content"], ratio, model_path, device_map)}
        for m in messages
    ]
