"""
============================================================
Prompts module — Gemini prompt templates for GoEmotions
============================================================

Provides:
    - EMOTION_LIST: 28 GoEmotions class names
    - ZERO_SHOT_TEMPLATE: zero-shot instruction template
    - FEW_SHOT_EXAMPLES: 5 diverse labelled examples
    - FEW_SHOT_TEMPLATE: few-shot template (examples + instruction)
    - build_prompt(): Assemble final prompt string
"""

from typing import List

# ============================================================
# Emotion list (28 classes — GoEmotions simplified)
# ============================================================
EMOTION_LIST: List[str] = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

# ============================================================
# Zero-shot template
# ============================================================
ZERO_SHOT_TEMPLATE = """\
You are an expert emotion analyst. Given a short text (typically from Reddit), \
identify ALL emotions expressed by the author.

RULES:
1. Select ALL emotions that apply — this is multi-label (a text can have zero, one, or more emotions).
2. Only use emotion labels from the list below. Do NOT invent new labels.
3. If no emotion is clearly expressed, use ["neutral"].
4. Output ONLY valid JSON — no explanation, no markdown, no extra text.

EMOTION LIST (28 classes):
{emotion_list}

OUTPUT FORMAT:
{{"emotions": ["emotion1", "emotion2"]}}

TEXT TO CLASSIFY:
"{text}"
"""

# ============================================================
# Few-shot examples (5 diverse samples)
# Covers: positive, negative, neutral, multi-label, rare class
# ============================================================
FEW_SHOT_EXAMPLES = [
    {
        "text": "I can't believe how amazing this turned out! I'm so proud of myself!",
        "emotions": ["excitement", "pride", "joy"],
        "note": "positive multi-label",
    },
    {
        "text": "This is honestly the worst thing I've ever seen. Absolutely disgusting.",
        "emotions": ["anger", "disgust", "disapproval"],
        "note": "negative multi-label",
    },
    {
        "text": "The meeting is scheduled for Tuesday at 3pm.",
        "emotions": ["neutral"],
        "note": "neutral single-label",
    },
    {
        "text": "I just found out my dog passed away this morning. I miss him so much.",
        "emotions": ["grief", "sadness"],
        "note": "rare class (grief) + sadness",
    },
    {
        "text": "Wait, really? I didn't see that coming at all — how did that happen?",
        "emotions": ["surprise", "curiosity"],
        "note": "surprise + curiosity",
    },
]

# ============================================================
# Few-shot template
# ============================================================
_FEW_SHOT_EXAMPLES_FORMATTED = "\n\n".join(
    f'TEXT: "{ex["text"]}"\nOUTPUT: {{"emotions": {ex["emotions"]}}}'
    for ex in FEW_SHOT_EXAMPLES
)

FEW_SHOT_TEMPLATE = """\
You are an expert emotion analyst. Given a short text (typically from Reddit), \
identify ALL emotions expressed by the author.

RULES:
1. Select ALL emotions that apply — this is multi-label (a text can have zero, one, or more emotions).
2. Only use emotion labels from the list below. Do NOT invent new labels.
3. If no emotion is clearly expressed, use ["neutral"].
4. Output ONLY valid JSON — no explanation, no markdown, no extra text.

EMOTION LIST (28 classes):
{emotion_list}

EXAMPLES:
{examples}

OUTPUT FORMAT:
{{"emotions": ["emotion1", "emotion2"]}}

TEXT TO CLASSIFY:
"{text}"
"""


# ============================================================
# Builder function
# ============================================================
def build_prompt(text: str, mode: str = "zero_shot") -> str:
    """
    Assemble the final prompt string for Gemini.

    Args:
        text: Input text to classify
        mode: "zero_shot" or "few_shot"

    Returns:
        Formatted prompt string ready to send to the Gemini API.

    Raises:
        ValueError: if mode is not recognised
    """
    emotion_list_str = "\n".join(f"- {e}" for e in EMOTION_LIST)

    if mode == "zero_shot":
        return ZERO_SHOT_TEMPLATE.format(
            emotion_list=emotion_list_str,
            text=text,
        )
    elif mode == "few_shot":
        return FEW_SHOT_TEMPLATE.format(
            emotion_list=emotion_list_str,
            examples=_FEW_SHOT_EXAMPLES_FORMATTED,
            text=text,
        )
    else:
        raise ValueError(f"Unknown prompt mode: '{mode}'. Choose 'zero_shot' or 'few_shot'.")
