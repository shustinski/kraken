from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np
import pytest

from model.tic_tac_toe_ai import (
    DRAW,
    TicTacToeNeuralOpponent,
    _softmax,
    available_moves,
    check_winner,
    encode_board,
    valid_move_mask,
)

pytest.importorskip("torch")


def _make_temp_state_path() -> Path:
    base = Path(".codex_tmp_scratch")
    base.mkdir(parents=True, exist_ok=True)
    temp_dir = base / f"ttt_ai_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / "ttt_state.json"


def test_check_winner_and_draw_states() -> None:
    assert check_winner(["X", "X", "X", "", "", "", "", "", ""]) == "X"
    assert check_winner(["O", "", "", "O", "", "", "O", "", ""]) == "O"
    assert check_winner(["X", "O", "X", "X", "O", "O", "O", "X", "X"]) == DRAW
    assert check_winner(["X", "", "", "", "", "", "", "", ""]) is None


def test_choose_move_is_always_legal() -> None:
    state_path = _make_temp_state_path()
    opponent = TicTacToeNeuralOpponent(state_path=state_path, exploration=0.0, seed=7)
    board = ["X", "O", "", "", "X", "", "", "O", ""]

    move = opponent.choose_move(board)
    assert move in available_moves(board)


def test_choose_move_finds_immediate_win_and_block() -> None:
    state_path = _make_temp_state_path()
    opponent = TicTacToeNeuralOpponent(state_path=state_path, exploration=0.0, seed=11)

    board_win = ["O", "O", "", "X", "X", "", "", "", ""]
    assert opponent.choose_move(board_win, ai_mark="O", user_mark="X") == 2

    board_block = ["X", "X", "", "O", "", "", "", "", ""]
    assert opponent.choose_move(board_block, ai_mark="O", user_mark="X") == 2


def test_positive_reward_increases_action_probability() -> None:
    state_path = _make_temp_state_path()
    opponent = TicTacToeNeuralOpponent(
        state_path=state_path,
        exploration=0.0,
        learning_rate=0.15,
        discount=1.0,
        seed=5,
    )
    board = ["X", "", "O", "", "", "", "", "", ""]
    action = 4
    step = opponent.make_policy_step(board, action)

    state = encode_board(board)
    mask = valid_move_mask(board)
    _, logits_before = opponent._forward(state)
    prob_before = float(_softmax(logits_before, mask)[action])

    for _ in range(30):
        opponent.learn_from_episode([step], reward=1.0)

    _, logits_after = opponent._forward(state)
    prob_after = float(_softmax(logits_after, mask)[action])
    assert prob_after > prob_before


def test_negative_reward_decreases_action_probability() -> None:
    state_path = _make_temp_state_path()
    opponent = TicTacToeNeuralOpponent(
        state_path=state_path,
        exploration=0.0,
        learning_rate=0.15,
        discount=1.0,
        seed=5,
    )
    board = ["X", "", "O", "", "", "", "", "", ""]
    action = 4
    step = opponent.make_policy_step(board, action)

    state = encode_board(board)
    mask = valid_move_mask(board)
    _, logits_before = opponent._forward(state)
    prob_before = float(_softmax(logits_before, mask)[action])

    for _ in range(30):
        opponent.learn_from_episode([step], reward=-1.0)

    _, logits_after = opponent._forward(state)
    prob_after = float(_softmax(logits_after, mask)[action])
    assert prob_after < prob_before


def test_state_persistence_restores_weights_and_stats() -> None:
    state_path = _make_temp_state_path()
    opponent = TicTacToeNeuralOpponent(state_path=state_path, exploration=0.0, seed=13)
    board = ["X", "", "", "", "", "", "", "", ""]
    step = opponent.make_policy_step(board, action=4)
    opponent.learn_from_episode([step], reward=1.0)
    opponent.register_game([step], winner="O", ai_mark="O")

    assert state_path.exists()
    saved_w1 = np.array(opponent.W1, copy=True)
    saved_stats = opponent.get_stats()

    reloaded = TicTacToeNeuralOpponent(state_path=state_path, exploration=0.0, seed=999)
    assert np.allclose(saved_w1, reloaded.W1)
    assert saved_stats == reloaded.get_stats()
