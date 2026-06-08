"""
============================================================
Prompts module — Load và build prompts cho LLM inference
============================================================

Usage:
    from src.prompts import build_prompt, VALID_EMOTIONS

    prompt = build_prompt("I love this!", mode="zero_shot")
    prompt = build_prompt("I love this!", mode="few_shot")
"""

from pathlib import Path

# ============================================================
# Constants
# ============================================================
PROMPTS_DIR = Path(__file__).parent

VALID_EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

_TEMPLATE_CACHE: dict = {}


def _load_template(mode: str) -> str:
    """Load template file, cache in memory."""
    if mode not in _TEMPLATE_CACHE:
        filename = f"{mode.replace('_', '')}_template.txt"
        path = PROMPTS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        _TEMPLATE_CACHE[mode] = path.read_text(encoding="utf-8")
    return _TEMPLATE_CACHE[mode]


def build_prompt(text: str, mode: str = "zero_shot") -> str:
    """
    Build prompt cho LLM inference.

    Args:
        text: Input text cần classify
        mode: "zero_shot" hoặc "few_shot"

    Returns:
        Prompt string đã điền text
    """
    if mode not in ("zero_shot", "few_shot"):
        raise ValueError(f"mode phải là 'zero_shot' hoặc 'few_shot', nhận được: {mode!r}")

    template = _load_template(mode)
    return template.replace('"{text}"', f'"{text}"')


def parse_response(response_text: str) -> list[str]:
    """
    Parse JSON response từ LLM, trả về list emotion names.

    Handles các format phổ biến từ open-weight models:
      - {"emotions": ["joy", "sadness"]}   (dict, expected format)
      - ["joy", "sadness"]                 (plain array)
      - Extra text surrounding JSON        (model thêm chú thích)
      - Markdown fences ```json ... ```

    Returns:
        List emotion names đã validate. Trả về ["neutral"] nếu parse fail.
    """
    import json
    import re

    if not response_text:
        return ["neutral"]

    # Strip markdown fences
    clean = re.sub(r"```(?:json)?\s*|```", "", response_text).strip()

    def _extract(obj) -> list[str]:
        if isinstance(obj, dict):
            emotions = obj.get("emotions", [])
        elif isinstance(obj, list):
            emotions = obj
        else:
            return []
        return [e for e in emotions if e in VALID_EMOTIONS]

    # Strategy 1: parse toàn bộ string
    try:
        valid = _extract(json.loads(clean))
        if valid:
            return valid
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: tìm JSON object {...} đầu tiên trong text
    m = re.search(r'\{.*?\}', clean, re.DOTALL)
    if m:
        try:
            valid = _extract(json.loads(m.group()))
            if valid:
                return valid
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: tìm JSON array [...] đầu tiên trong text
    m = re.search(r'\[.*?\]', clean, re.DOTALL)
    if m:
        try:
            valid = _extract(json.loads(m.group()))
            if valid:
                return valid
        except (json.JSONDecodeError, ValueError):
            pass

    return ["neutral"]
