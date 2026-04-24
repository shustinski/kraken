from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception:  # pragma: no cover - runtime-only fallback when torch is unavailable.
    torch = None
    nn = None
    F = None


Board = Sequence[str]

PLAYER_X = "X"
PLAYER_O = "O"
DRAW = "draw"

_MASKED_LOGIT_VALUE = -1.0e9
_MASKED_Q_VALUE = -1.0e6
_POSITIONAL_BIAS = np.asarray(
    [
        0.06,
        0.01,
        0.06,
        0.01,
        0.16,
        0.01,
        0.06,
        0.01,
        0.06,
    ],
    dtype=np.float32,
)


def _build_symmetry_maps() -> list[tuple[np.ndarray, np.ndarray]]:
    index_grid = np.arange(9, dtype=np.int64).reshape(3, 3)
    transforms = [
        lambda m: m,
        lambda m: np.rot90(m, 1),
        lambda m: np.rot90(m, 2),
        lambda m: np.rot90(m, 3),
        np.fliplr,
        np.flipud,
        np.transpose,
        lambda m: np.fliplr(np.transpose(m)),
    ]
    maps: list[tuple[np.ndarray, np.ndarray]] = []
    for transform in transforms:
        old_at_new = np.asarray(transform(index_grid), dtype=np.int64).reshape(9)
        old_to_new = np.empty(9, dtype=np.int64)
        for new_idx, old_idx in enumerate(old_at_new):
            old_to_new[int(old_idx)] = int(new_idx)
        maps.append((old_at_new, old_to_new))
    return maps


_SYMMETRY_MAPS = _build_symmetry_maps()


@dataclass(frozen=True)
class PolicyStep:
    state: np.ndarray
    valid_mask: np.ndarray
    action: int


def available_moves(board: Board) -> list[int]:
    return [idx for idx, cell in enumerate(board) if not cell]


def check_winner(board: Board) -> str | None:
    wins = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    )
    for a, b, c in wins:
        if board[a] and board[a] == board[b] == board[c]:
            return str(board[a])
    if all(board):
        return DRAW
    return None


def encode_board(board: Board, ai_mark: str = PLAYER_O, user_mark: str = PLAYER_X) -> np.ndarray:
    vector = np.zeros(9, dtype=np.float32)
    for idx, cell in enumerate(board):
        if cell == ai_mark:
            vector[idx] = 1.0
        elif cell == user_mark:
            vector[idx] = -1.0
    return vector


def valid_move_mask(board: Board) -> np.ndarray:
    return np.asarray([not cell for cell in board], dtype=bool)


def _softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not bool(np.any(mask)):
        return np.zeros_like(logits, dtype=np.float32)
    masked_logits = np.where(mask, logits, _MASKED_LOGIT_VALUE)
    shifted = masked_logits - np.max(masked_logits[mask])
    exps = np.zeros_like(logits, dtype=np.float32)
    exps[mask] = np.exp(shifted[mask]).astype(np.float32)
    total = float(np.sum(exps[mask]))
    if total <= 0.0:
        fallback = np.zeros_like(logits, dtype=np.float32)
        valid_count = int(np.count_nonzero(mask))
        if valid_count > 0:
            fallback[mask] = 1.0 / valid_count
        return fallback
    return exps / total


if nn is not None:
    class _TicTacToeQNetwork(nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(9, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 9)
            self.activation = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            hidden = self.activation(self.fc1(x))
            return self.fc2(hidden)
else:
    class _TicTacToeQNetwork:  # pragma: no cover
        pass


class TicTacToeNeuralOpponent:
    def __init__(
        self,
        state_path: Path | None = None,
        *,
        hidden_size: int = 128,
        learning_rate: float = 1.0e-3,
        exploration: float = 0.12,
        min_exploration: float = 0.02,
        exploration_decay: float = 0.995,
        discount: float = 0.95,
        replay_capacity: int = 12000,
        batch_size: int = 96,
        training_updates_per_game: int = 48,
        seed: int | None = None,
    ) -> None:
        if torch is None or nn is None or F is None:
            raise RuntimeError(
                "Для мини-игры нужен PyTorch. Установите torch и перезапустите приложение."
            )

        self.hidden_size = max(16, int(hidden_size))
        self.learning_rate = max(1.0e-5, float(learning_rate))
        self.exploration = min(1.0, max(0.0, float(exploration)))
        self.min_exploration = min(1.0, max(0.0, float(min_exploration)))
        if self.min_exploration > self.exploration:
            self.min_exploration = self.exploration
        self.exploration_decay = min(1.0, max(0.8, float(exploration_decay)))
        self.discount = min(1.0, max(0.0, float(discount)))
        self.replay_capacity = max(256, int(replay_capacity))
        self.batch_size = max(16, int(batch_size))
        self.training_updates_per_game = max(8, int(training_updates_per_game))
        self._initial_exploration = float(self.exploration)

        self._rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(int(seed))

        self.state_path = self._resolve_state_path(state_path)
        self._device = torch.device("cpu")
        self._model = _TicTacToeQNetwork(self.hidden_size).to(self._device)
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=1.0e-4,
        )
        self._reset_network_weights()

        self._replay_states: list[np.ndarray] = []
        self._replay_masks: list[np.ndarray] = []
        self._replay_actions: list[int] = []
        self._replay_targets: list[float] = []
        self._replay_position = 0

        self.games_total = 0
        self.ai_wins = 0
        self.user_wins = 0
        self.draws = 0

        self.load_state()
        self._model.eval()

    @property
    def W1(self) -> np.ndarray:
        return self._model.fc1.weight.detach().cpu().numpy().T.copy()

    @property
    def b1(self) -> np.ndarray:
        return self._model.fc1.bias.detach().cpu().numpy().copy()

    @property
    def W2(self) -> np.ndarray:
        return self._model.fc2.weight.detach().cpu().numpy().T.copy()

    @property
    def b2(self) -> np.ndarray:
        return self._model.fc2.bias.detach().cpu().numpy().copy()

    @staticmethod
    def _resolve_state_path(state_path: Path | None) -> Path:
        if state_path is not None:
            return Path(state_path)
        settings_root = os.getenv("NEURALIMAGE_SETTINGS_DIR")
        if settings_root:
            return Path(settings_root) / "tic_tac_toe_ai_state.json"
        return Path.cwd() / "tic_tac_toe_ai_state.json"

    def _forward(self, state_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state_vector, dtype=np.float32).reshape(1, 9)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            hidden = self._model.activation(self._model.fc1(state_tensor))
            logits = self._model.fc2(hidden)
        return (
            hidden.squeeze(0).cpu().numpy().astype(np.float32),
            logits.squeeze(0).cpu().numpy().astype(np.float32),
        )

    def choose_move(
        self,
        board: Board,
        *,
        ai_mark: str = PLAYER_O,
        user_mark: str = PLAYER_X,
    ) -> int:
        moves = available_moves(board)
        if not moves:
            return -1

        winning_move = self._find_forced_move(board, ai_mark)
        if winning_move >= 0:
            return winning_move

        blocking_move = self._find_forced_move(board, user_mark)
        if blocking_move >= 0:
            return blocking_move

        if self._rng.random() < self.exploration:
            return int(self._rng.choice(moves))

        state = encode_board(board, ai_mark=ai_mark, user_mark=user_mark)
        mask = valid_move_mask(board)
        q_values = self._predict_q_values(state)
        legal_scores = np.where(mask, q_values + _POSITIONAL_BIAS, _MASKED_LOGIT_VALUE)
        best_score = float(np.max(legal_scores[mask]))
        best_moves = [move for move in moves if abs(float(legal_scores[move]) - best_score) < 1.0e-6]
        return int(self._rng.choice(best_moves))

    def make_policy_step(
        self,
        board_before_move: Board,
        action: int,
        *,
        ai_mark: str = PLAYER_O,
        user_mark: str = PLAYER_X,
    ) -> PolicyStep:
        return PolicyStep(
            state=encode_board(board_before_move, ai_mark=ai_mark, user_mark=user_mark),
            valid_mask=valid_move_mask(board_before_move),
            action=int(action),
        )

    def register_game(self, steps: Sequence[PolicyStep], winner: str, ai_mark: str = PLAYER_O) -> None:
        outcome = winner
        self.games_total += 1
        if outcome == ai_mark:
            self.ai_wins += 1
            reward = 1.0
            self.exploration = max(self.min_exploration, self.exploration * self.exploration_decay)
        elif outcome == DRAW:
            self.draws += 1
            reward = 0.35
            self.exploration = max(self.min_exploration, self.exploration * self.exploration_decay)
        else:
            self.user_wins += 1
            reward = -1.0
            self.exploration = min(1.0, self.exploration * (2.0 - self.exploration_decay))
        self.learn_from_episode(steps, reward=reward)
        self.save_state()

    def learn_from_episode(self, steps: Sequence[PolicyStep], *, reward: float) -> None:
        if not steps:
            return
        base_reward = float(reward)
        for step_index, step in enumerate(steps):
            discounted_reward = base_reward * (self.discount ** (len(steps) - step_index - 1))
            self._store_augmented_step(step, discounted_reward)
        updates = min(192, self.training_updates_per_game + len(steps) * 6)
        self._train_from_replay(updates)

    def _gradient_step(self, step: PolicyStep, reward: float) -> None:
        self.learn_from_episode([step], reward=reward)

    def reset_learning(self) -> None:
        self._reset_network_weights()
        self._replay_states.clear()
        self._replay_masks.clear()
        self._replay_actions.clear()
        self._replay_targets.clear()
        self._replay_position = 0
        self.games_total = 0
        self.ai_wins = 0
        self.user_wins = 0
        self.draws = 0
        self.exploration = self._initial_exploration
        self.save_state()

    def get_stats(self) -> dict[str, int]:
        return {
            "games_total": int(self.games_total),
            "ai_wins": int(self.ai_wins),
            "user_wins": int(self.user_wins),
            "draws": int(self.draws),
        }

    def save_state(self) -> None:
        payload = {
            "format_version": 2,
            "hidden_size": int(self.hidden_size),
            "learning_rate": float(self.learning_rate),
            "exploration": float(self.exploration),
            "min_exploration": float(self.min_exploration),
            "exploration_decay": float(self.exploration_decay),
            "discount": float(self.discount),
            "games_total": int(self.games_total),
            "ai_wins": int(self.ai_wins),
            "user_wins": int(self.user_wins),
            "draws": int(self.draws),
            "model_state": self._serialize_model_state(),
            # Keep legacy keys for backward compatibility with old checkpoints.
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)

    def load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            with self.state_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(payload, dict):
            return

        hidden_size = int(payload.get("hidden_size", self.hidden_size))
        if hidden_size != self.hidden_size:
            return

        self.learning_rate = max(1.0e-5, float(payload.get("learning_rate", self.learning_rate)))
        self.exploration = min(1.0, max(0.0, float(payload.get("exploration", self.exploration))))
        self.min_exploration = min(1.0, max(0.0, float(payload.get("min_exploration", self.min_exploration))))
        if self.min_exploration > self.exploration:
            self.min_exploration = self.exploration
        self.exploration_decay = min(1.0, max(0.8, float(payload.get("exploration_decay", self.exploration_decay))))
        self.discount = min(1.0, max(0.0, float(payload.get("discount", self.discount))))
        self._initial_exploration = float(self.exploration)

        loaded = False
        model_state = payload.get("model_state")
        if isinstance(model_state, dict):
            loaded = self._load_model_state_dict(model_state)
        if not loaded:
            loaded = self._load_legacy_weights(payload)
        if loaded:
            self._sync_optimizer_lr()

        self.games_total = int(payload.get("games_total", self.games_total))
        self.ai_wins = int(payload.get("ai_wins", self.ai_wins))
        self.user_wins = int(payload.get("user_wins", self.user_wins))
        self.draws = int(payload.get("draws", self.draws))

    def _serialize_model_state(self) -> dict[str, list]:
        state = self._model.state_dict()
        return {name: tensor.detach().cpu().numpy().tolist() for name, tensor in state.items()}

    def _load_model_state_dict(self, raw_state: dict) -> bool:
        current_state = self._model.state_dict()
        converted: dict[str, torch.Tensor] = {}
        for name, ref_tensor in current_state.items():
            raw_tensor = raw_state.get(name)
            if raw_tensor is None:
                return False
            arr = np.asarray(raw_tensor, dtype=np.float32)
            if arr.shape != tuple(ref_tensor.shape):
                return False
            converted[name] = torch.from_numpy(arr)
        self._model.load_state_dict(converted, strict=True)
        return True

    def _load_legacy_weights(self, payload: dict) -> bool:
        try:
            w1 = np.asarray(payload.get("W1"), dtype=np.float32)
            b1 = np.asarray(payload.get("b1"), dtype=np.float32)
            w2 = np.asarray(payload.get("W2"), dtype=np.float32)
            b2 = np.asarray(payload.get("b2"), dtype=np.float32)
        except (TypeError, ValueError):
            return False
        if w1.shape != (9, self.hidden_size) or w2.shape != (self.hidden_size, 9):
            return False
        if b1.shape != (self.hidden_size,) or b2.shape != (9,):
            return False

        with torch.no_grad():
            self._model.fc1.weight.copy_(torch.from_numpy(w1.T))
            self._model.fc1.bias.copy_(torch.from_numpy(b1))
            self._model.fc2.weight.copy_(torch.from_numpy(w2.T))
            self._model.fc2.bias.copy_(torch.from_numpy(b2))
        return True

    def _sync_optimizer_lr(self) -> None:
        for group in self._optimizer.param_groups:
            group["lr"] = self.learning_rate

    def _reset_network_weights(self) -> None:
        with torch.no_grad():
            scale1 = np.sqrt(2.0 / 9.0)
            scale2 = np.sqrt(2.0 / float(self.hidden_size))
            w1 = self._rng.normal(0.0, scale1, size=(self.hidden_size, 9)).astype(np.float32)
            b1 = np.zeros(self.hidden_size, dtype=np.float32)
            w2 = self._rng.normal(0.0, scale2, size=(9, self.hidden_size)).astype(np.float32)
            b2 = np.zeros(9, dtype=np.float32)
            self._model.fc1.weight.copy_(torch.from_numpy(w1))
            self._model.fc1.bias.copy_(torch.from_numpy(b1))
            self._model.fc2.weight.copy_(torch.from_numpy(w2))
            self._model.fc2.bias.copy_(torch.from_numpy(b2))

    def _find_forced_move(self, board: Board, mark: str) -> int:
        for move in available_moves(board):
            candidate = list(board)
            candidate[move] = mark
            if check_winner(candidate) == mark:
                return move
        return -1

    def _predict_q_values(self, state_vector: np.ndarray) -> np.ndarray:
        state = np.asarray(state_vector, dtype=np.float32).reshape(1, 9)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            q_values = self._model(state_tensor).squeeze(0)
        return q_values.cpu().numpy().astype(np.float32)

    def _store_augmented_step(self, step: PolicyStep, target_value: float) -> None:
        action = int(step.action)
        if action < 0 or action >= 9:
            return
        if not bool(step.valid_mask[action]):
            return
        for old_at_new, old_to_new in _SYMMETRY_MAPS:
            augmented_state = np.asarray(step.state[old_at_new], dtype=np.float32)
            augmented_mask = np.asarray(step.valid_mask[old_at_new], dtype=bool)
            augmented_action = int(old_to_new[action])
            self._push_replay(augmented_state, augmented_mask, augmented_action, float(target_value))

    def _push_replay(self, state: np.ndarray, mask: np.ndarray, action: int, target_value: float) -> None:
        if len(self._replay_states) < self.replay_capacity:
            self._replay_states.append(np.asarray(state, dtype=np.float32))
            self._replay_masks.append(np.asarray(mask, dtype=bool))
            self._replay_actions.append(int(action))
            self._replay_targets.append(float(target_value))
            return

        pos = self._replay_position
        self._replay_states[pos] = np.asarray(state, dtype=np.float32)
        self._replay_masks[pos] = np.asarray(mask, dtype=bool)
        self._replay_actions[pos] = int(action)
        self._replay_targets[pos] = float(target_value)
        self._replay_position = (self._replay_position + 1) % self.replay_capacity

    def _train_from_replay(self, updates: int) -> None:
        replay_size = len(self._replay_states)
        if replay_size == 0:
            return

        self._model.train()
        for _ in range(max(1, int(updates))):
            batch_size = min(self.batch_size, replay_size)
            indices = self._rng.choice(replay_size, size=batch_size, replace=False)

            states = np.stack([self._replay_states[int(i)] for i in indices]).astype(np.float32)
            masks = np.stack([self._replay_masks[int(i)] for i in indices]).astype(bool)
            actions = np.asarray([self._replay_actions[int(i)] for i in indices], dtype=np.int64)
            targets = np.asarray([self._replay_targets[int(i)] for i in indices], dtype=np.float32)

            states_t = torch.as_tensor(states, dtype=torch.float32, device=self._device)
            masks_t = torch.as_tensor(masks, dtype=torch.bool, device=self._device)
            actions_t = torch.as_tensor(actions, dtype=torch.long, device=self._device)
            targets_t = torch.as_tensor(targets, dtype=torch.float32, device=self._device)

            action_is_legal = masks_t.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            if not bool(torch.any(action_is_legal)):
                continue
            if not bool(torch.all(action_is_legal)):
                legal_indices = torch.nonzero(action_is_legal, as_tuple=False).squeeze(1)
                states_t = states_t.index_select(0, legal_indices)
                masks_t = masks_t.index_select(0, legal_indices)
                actions_t = actions_t.index_select(0, legal_indices)
                targets_t = targets_t.index_select(0, legal_indices)

            q_values = self._model(states_t)
            masked_q = q_values.masked_fill(~masks_t, _MASKED_Q_VALUE)
            chosen_q = masked_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            td_loss = F.smooth_l1_loss(chosen_q, targets_t)

            other_q = masked_q.clone()
            other_q.scatter_(1, actions_t.unsqueeze(1), _MASKED_Q_VALUE)
            strongest_other = other_q.max(dim=1).values
            has_alternative = (masks_t.sum(dim=1) > 1).float()
            positive_target = (targets_t > 0).float()
            margin_mask = has_alternative * positive_target
            if bool(torch.any(margin_mask > 0)):
                margin = F.relu(0.2 - (chosen_q - strongest_other))
                margin_loss = (margin * margin_mask).sum() / margin_mask.sum()
            else:
                margin_loss = torch.zeros((), dtype=torch.float32, device=self._device)

            loss = td_loss + 0.2 * margin_loss

            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optimizer.step()
        self._model.eval()
