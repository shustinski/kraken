from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from neuralimage.lib.ui_texts import get_ui_section
from neuralimage.model.tic_tac_toe_ai import (
    DRAW,
    PLAYER_O,
    PLAYER_X,
    TicTacToeNeuralOpponent,
    available_moves,
    check_winner,
)


class TicTacToeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._texts = get_ui_section("tic_tac_toe")
        self._ai: TicTacToeNeuralOpponent | None = None
        self._ai_error: str = ""
        try:
            self._ai = TicTacToeNeuralOpponent()
        except RuntimeError as exc:
            self._ai_error = str(exc)
        self._board = [""] * 9
        self._episode_steps = []
        self._game_over = False

        self.user_mark = PLAYER_X
        self.ai_mark = PLAYER_O

        self._setup_ui()
        if self._ai is None:
            self._set_unavailable_state()
        else:
            self._sync_stats()
            self._start_new_game()

    def _setup_ui(self) -> None:
        title = self._texts.get("window_title", "Крестики-нолики: нейросеть")
        self.setWindowTitle(str(title))
        self.setMinimumWidth(360)
        self.setModal(False)

        root = QVBoxLayout(self)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        board_widget = QWidget(self)
        board_layout = QGridLayout(board_widget)
        board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.setSpacing(6)
        self.cell_buttons: list[QPushButton] = []
        for idx in range(9):
            button = QPushButton("")
            button.setFixedSize(92, 92)
            button.setStyleSheet("font-size: 26px; font-weight: bold;")
            button.clicked.connect(self._build_cell_handler(idx))
            self.cell_buttons.append(button)
            board_layout.addWidget(button, idx // 3, idx % 3)
        root.addWidget(board_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        root.addWidget(self.stats_label)

        controls = QHBoxLayout()
        self.ai_starts_checkbox = QCheckBox(str(self._texts.get("ai_starts", "ИИ ходит первым")))
        controls.addWidget(self.ai_starts_checkbox)

        self.new_game_button = QPushButton(str(self._texts.get("new_game", "Новая партия")))
        self.new_game_button.clicked.connect(self._start_new_game)
        controls.addWidget(self.new_game_button)

        self.reset_ai_button = QPushButton(str(self._texts.get("reset_ai", "Сбросить обучение ИИ")))
        self.reset_ai_button.clicked.connect(self._reset_ai)
        controls.addWidget(self.reset_ai_button)
        root.addLayout(controls)

    def _build_cell_handler(self, index: int) -> Callable[[], None]:
        return lambda: self._on_user_move(index)

    def _start_new_game(self) -> None:
        self._board = [""] * 9
        self._episode_steps = []
        self._game_over = False
        for button in self.cell_buttons:
            button.setEnabled(True)
            button.setText("")

        if self.ai_starts_checkbox.isChecked():
            self._set_status("ai_thinking")
            self._perform_ai_move()
        else:
            self._set_status("user_turn")

    def _reset_ai(self) -> None:
        if self._ai is None:
            return
        confirmation = QMessageBox.question(
            self,
            str(self._texts.get("reset_title", "Сброс обучения")),
            str(self._texts.get("reset_confirm", "Удалить накопленный игровой опыт ИИ?")),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmation != QMessageBox.StandardButton.Yes:
            return
        self._ai.reset_learning()
        self._sync_stats()
        self._start_new_game()

    def _on_user_move(self, index: int) -> None:
        if self._game_over or self._board[index]:
            return
        self._board[index] = self.user_mark
        self._refresh_board()
        if self._finish_if_game_over():
            return
        self._set_status("ai_thinking")
        self._perform_ai_move()

    def _perform_ai_move(self) -> None:
        if self._ai is None:
            return
        if self._game_over:
            return
        moves = available_moves(self._board)
        if not moves:
            self._finish_if_game_over()
            return
        move = self._ai.choose_move(self._board, ai_mark=self.ai_mark, user_mark=self.user_mark)
        if move < 0 or move not in moves:
            move = moves[0]
        step = self._ai.make_policy_step(self._board, move, ai_mark=self.ai_mark, user_mark=self.user_mark)
        self._episode_steps.append(step)
        self._board[move] = self.ai_mark
        self._refresh_board()
        if not self._finish_if_game_over():
            self._set_status("user_turn")

    def _finish_if_game_over(self) -> bool:
        if self._ai is None:
            return False
        winner = check_winner(self._board)
        if winner is None:
            return False
        self._game_over = True
        for button in self.cell_buttons:
            button.setEnabled(False)

        self._ai.register_game(self._episode_steps, winner, ai_mark=self.ai_mark)
        self._sync_stats()

        if winner == self.user_mark:
            self._set_status("user_win")
        elif winner == self.ai_mark:
            self._set_status("ai_win")
        elif winner == DRAW:
            self._set_status("draw")
        else:
            self._set_status("finished")
        return True

    def _refresh_board(self) -> None:
        for idx, button in enumerate(self.cell_buttons):
            button.setText(self._board[idx])
            button.setEnabled(not self._game_over and not bool(self._board[idx]))

    def _sync_stats(self) -> None:
        if self._ai is None:
            return
        stats = self._ai.get_stats()
        text = str(
            self._texts.get(
                "stats_template",
                "Партии: {games} | Победы ИИ: {ai_wins} | Победы игрока: {user_wins} | Ничьи: {draws}",
            )
        ).format(
            games=stats["games_total"],
            ai_wins=stats["ai_wins"],
            user_wins=stats["user_wins"],
            draws=stats["draws"],
        )
        self.stats_label.setText(text)

    def _set_status(self, key: str) -> None:
        fallback = {
            "user_turn": "Ваш ход (X).",
            "ai_thinking": "Ход ИИ (O)...",
            "user_win": "Вы выиграли. Нейросеть учится на этой партии.",
            "ai_win": "Победил ИИ. Его веса обновлены.",
            "draw": "Ничья. Нейросеть запомнила результат.",
            "finished": "Партия завершена.",
        }
        text = self._texts.get(f"status_{key}", fallback.get(key, ""))
        self.status_label.setText(str(text))

    def _set_unavailable_state(self) -> None:
        message = self._ai_error or str(
            self._texts.get("status_unavailable", "Мини-игра недоступна: не найден PyTorch.")
        )
        self.status_label.setText(message)
        for button in self.cell_buttons:
            button.setEnabled(False)
        self.ai_starts_checkbox.setEnabled(False)
        self.new_game_button.setEnabled(False)
        self.reset_ai_button.setEnabled(False)
        self.stats_label.setText("—")
