"""Constrained detection of an @board invocation in recognized handwriting."""
import re
from typing import Dict


# OCR commonly confuses the last letters in "board" (for example, "boart"
# or "boat") and may insert spaces between letters. Keep matching anchored to
# the start so ordinary mentions of a board never become commands.
_TRIGGER = re.compile(
    r"^\s*(?:@\s*b\s*[o0]\s*a\s*(?:r\s*)?[dtr]|"
    r"at\s+b\s*[o0]\s*a\s*(?:r\s*)?[dtr])\b\s*[,.:;\-]?\s*",
    re.IGNORECASE,
)


def extract_board_command(text: str) -> Dict[str, object]:
    """Recognize only an invocation at the start of the transcription.

    Accepts spaced characters and common OCR substitutions in ``@board`` and
    ``at board`` (including ``@boart`` and ``at boat``). An unrelated
    occurrence of the word board is deliberately not a trigger.
    """
    transcription = (text or "").strip()
    match = _TRIGGER.match(transcription)
    if not match:
        return {"command_detected": False, "command_text": None,
                "instruction_missing": False}
    instruction = transcription[match.end():].strip()
    return {"command_detected": True, "command_text": instruction or None,
            "instruction_missing": not bool(instruction)}
