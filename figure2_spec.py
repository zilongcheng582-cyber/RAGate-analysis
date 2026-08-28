"""Shared display specification for the paper Figure 2."""

PAPER_FIGURE2_ORDER = [
    "turn_position_ratio",
    "turn_position_squared",
    "user_has_question",
    "prev_sys_is_question",
    "user_starts_question_word",
    "user_turn_len_log",
    "sys_turn_len_log",
    "dialogue_len_log",
    "consecutive_sys_turns",
    "turn_len_ratio",
]

PAPER_FIGURE2_LABELS = {
    "turn_position_ratio": "pos_ratio",
    "turn_position_squared": "pos_sq",
    "user_has_question": "user_q",
    "prev_sys_is_question": "prev_sys_q",
    "user_starts_question_word": "user_q_word",
    "user_turn_len_log": "usr_len",
    "sys_turn_len_log": "sys_len",
    "dialogue_len_log": "dlg_len",
    "consecutive_sys_turns": "cons_sys",
    "turn_len_ratio": "len_ratio",
}
