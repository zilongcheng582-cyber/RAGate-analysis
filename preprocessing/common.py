"""Shared deterministic serialization and feature helpers for preprocessing."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


QUESTION_WORDS = {"what", "how", "where", "when", "why", "is", "does", "can", "do"}
INSTRUCTION = (
    "Analyse the conversational context so far. Estimate if augmenting the response "
    "with external knowledge is helpful with an output of 'True' or 'False' only."
)
FEATURE_COLUMNS = [
    "dialogue_id",
    "turn_idx",
    "turn_position_ratio",
    "prev_sys_is_question",
    "user_has_question",
    "user_starts_question_word",
    "user_turn_len_log",
    "sys_turn_len_log",
    "dialogue_len_log",
    "consecutive_sys_turns",
    "turn_len_ratio",
    "turn_position_squared",
    "label",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> int:
    """Write canonical CSV dialect: UTF-8, CRLF, minimal quoting, no BOM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="raise",
            lineterminator="\r\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
            count += 1
    return count


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)) and value in (0, 1):
        return "True" if int(value) else "False"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1"}:
            return "True"
        if text in {"false", "0"}:
            return "False"
    raise ValueError(f"Expected boolean label, got {value!r}")


def label_int(value: Any) -> int:
    return 1 if bool_text(value) == "True" else 0


def count_tokens(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text.lower())) if text else 0


def require_text(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context}: {field} must be a string, got {type(value).__name__}")
    return value


def serialize_turns(turns: Sequence[Mapping[str, Any]], speaker_map: Mapping[str, str]) -> str:
    serialized: list[str] = []
    for turn_index, turn in enumerate(turns):
        speaker = turn.get("speaker")
        if speaker not in speaker_map:
            raise ValueError(f"turn {turn_index}: unexpected speaker {speaker!r}")
        text = require_text(turn.get("text"), "text", f"turn {turn_index}")
        serialized.append(f"{speaker_map[speaker]}: {text}")
    if not serialized:
        raise ValueError("dialogue contains no turns")
    return " ".join(serialized)


def parse_processed_turns(input_text: str) -> list[dict[str, str]]:
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError("processed input is empty")
    parts = re.split(r"(USER:|SYSTEM:)", input_text.strip())
    turns: list[dict[str, str]] = []
    if len(parts) < 3 or parts[0] != "":
        raise ValueError(f"cannot parse processed input: {input_text[:80]!r}")
    i = 1
    while i < len(parts) - 1:
        speaker = parts[i].rstrip(":")
        text = parts[i + 1].strip()
        if speaker not in {"USER", "SYSTEM"} or not text:
            raise ValueError(f"malformed processed turn near {parts[i:i + 2]!r}")
        turns.append({"speaker": speaker, "text": text})
        i += 2
    if not turns or turns[-1]["speaker"] != "USER":
        raise ValueError("processed input must end at a USER turn")
    return turns


def _base_features(
    dialogue_id: Any,
    turn_idx: int,
    user_text: str,
    prev_sys_text: str,
    dialogue_len: int,
    consecutive_sys: int,
    label: Any,
) -> dict[str, Any]:
    user_turn_len = count_tokens(user_text)
    sys_turn_len = count_tokens(prev_sys_text)
    ratio = turn_idx / dialogue_len if dialogue_len > 0 else 0.0
    first_word = re.findall(r"\b\w+\b", user_text.lower())
    return {
        "dialogue_id": dialogue_id,
        "turn_idx": turn_idx,
        "turn_position_ratio": ratio,
        "prev_sys_is_question": 1 if prev_sys_text.strip().endswith("?") else 0,
        "user_has_question": 1 if "?" in user_text else 0,
        "user_starts_question_word": 1 if first_word and first_word[0] in QUESTION_WORDS else 0,
        "user_turn_len_log": math.log(1 + user_turn_len),
        "sys_turn_len_log": math.log(1 + sys_turn_len),
        "dialogue_len_log": math.log(dialogue_len) if dialogue_len > 1 else 0.0,
        "consecutive_sys_turns": consecutive_sys,
        "turn_len_ratio": min(user_turn_len / max(sys_turn_len, 1), 5.0),
        "turn_position_squared": ratio**2,
        "label": label_int(label),
    }


def ketod_feature_rows(dialogue: Mapping[str, Any]) -> list[dict[str, Any]]:
    turns = dialogue["turns"]
    if not isinstance(turns, list):
        raise ValueError(f"KETOD {dialogue.get('dialogue_id')}: turns must be a list")
    total_turns = len(turns)
    user_count = sum(1 for turn in turns if turn.get("speaker") == "USER")
    rows: list[dict[str, Any]] = []
    prev_sys_text = ""
    consecutive_sys = 0
    user_turn_idx = 0
    for index, turn in enumerate(turns):
        speaker = turn.get("speaker")
        if speaker == "USER":
            user_turn_idx += 1
            user_text = require_text(turn.get("utterance"), "utterance", f"KETOD turn {index}")
            if index + 1 < total_turns and turns[index + 1].get("speaker") == "SYSTEM":
                sys_turn = turns[index + 1]
                sys_text = require_text(sys_turn.get("utterance"), "utterance", f"KETOD turn {index + 1}")
                rows.append(
                    _base_features(
                        dialogue.get("dialogue_id"),
                        user_turn_idx,
                        user_text,
                        prev_sys_text,
                        user_count,
                        consecutive_sys,
                        sys_turn.get("enrich", False),
                    )
                )
                prev_sys_text = sys_text
            # This reset must happen after capturing the row; it is the audited KETOD fix.
            consecutive_sys = 0
        elif speaker == "SYSTEM":
            prev_sys_text = require_text(turn.get("utterance"), "utterance", f"KETOD turn {index}")
            consecutive_sys += 1
        else:
            raise ValueError(f"KETOD turn {index}: unexpected speaker {speaker!r}")
    return rows


def dstc9_feature_row(log: Sequence[Mapping[str, Any]], label: Any, dialogue_id: int) -> dict[str, Any]:
    turns = list(log)
    if not isinstance(label, Mapping) or "target" not in label:
        raise ValueError(f"DSTC9 dialogue {dialogue_id}: malformed label")
    user_turns = [turn for turn in turns if turn.get("speaker") == "U"]
    if not user_turns:
        raise ValueError(f"DSTC9 dialogue {dialogue_id}: no USER turn")
    dialogue_len = len(user_turns)
    user_turn_idx = dialogue_len
    user_text = require_text(user_turns[-1].get("text"), "text", f"DSTC9 dialogue {dialogue_id}")
    prev_sys_text = ""
    consecutive_sys = 0
    found_current_user = False
    for turn in reversed(turns):
        speaker = turn.get("speaker")
        if speaker == "U" and not found_current_user:
            found_current_user = True
            continue
        if found_current_user:
            if speaker == "S":
                text = require_text(turn.get("text"), "text", f"DSTC9 dialogue {dialogue_id}")
                if not prev_sys_text:
                    prev_sys_text = text
                consecutive_sys += 1
            else:
                break
    return _base_features(
        dialogue_id,
        user_turn_idx,
        user_text,
        prev_sys_text,
        dialogue_len,
        consecutive_sys,
        label["target"],
    )


def processed_feature_row(input_text: str, label: Any, dialogue_id: int) -> dict[str, Any]:
    turns = parse_processed_turns(input_text)
    user_turns = [turn for turn in turns if turn["speaker"] == "USER"]
    dialogue_len = len(user_turns)
    if not user_turns:
        raise ValueError(f"processed dialogue {dialogue_id}: no USER turn")
    user_text = user_turns[-1]["text"]
    prev_sys_text = ""
    consecutive_sys = 0
    found_current_user = False
    for turn in reversed(turns):
        if turn["speaker"] == "USER" and not found_current_user:
            found_current_user = True
            continue
        if found_current_user:
            if turn["speaker"] == "SYSTEM":
                if not prev_sys_text:
                    prev_sys_text = turn["text"]
                consecutive_sys += 1
            else:
                break
    return _base_features(dialogue_id, dialogue_len, user_text, prev_sys_text, dialogue_len, consecutive_sys, label)
