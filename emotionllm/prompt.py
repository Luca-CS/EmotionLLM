"""Detected emotion -> tone instruction for the language model."""

# Same mapping as the original project: the label picks the tone of the answer.
TONES = {
    "Angry": ("calm", "Stay calm and level-headed, acknowledge the frustration in one short sentence, never be defensive."),
    "Excited": ("enthusiastic", "Match the energy: be upbeat and enthusiastic."),
    "Frustrated": ("reassuring", "Be reassuring: acknowledge the difficulty, then give a clear and practical way forward."),
    "Happy": ("positive", "Be warm and positive."),
    "Neutral": ("neutral", ""),
    "Sad": ("empathetic", "Be gentle and empathetic: acknowledge the feeling first, do not rush to solutions."),
}

BASE_PROMPT = (
    "You are a voice assistant. Reply in the language of the user, in two to four short sentences of plain text "
    "(no lists, no markdown)."
)


def tone_for(emotion: str) -> str:
    return TONES.get(emotion, TONES["Neutral"])[0]


def system_prompt(emotion: str) -> str:
    """The base prompt, plus a tone instruction when the emotion is not neutral."""
    _, guidance = TONES.get(emotion, TONES["Neutral"])
    if not guidance:
        return BASE_PROMPT
    return (
        f"{BASE_PROMPT} The user's voice sounded {emotion.lower()}. {guidance} "
        "Do not tell the user which emotion you detected."
    )


def build_messages(transcript: str, emotion: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(emotion)},
        {"role": "user", "content": transcript.strip()},
    ]
