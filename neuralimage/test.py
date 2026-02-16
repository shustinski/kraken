# import datetime
# from time import sleep
#
# now = datetime.datetime.now()
# while True:
#         now_new = datetime.datetime.now()
#         print(now_new.strftime('%H:%M:%S'))
#         now_new = now_new - now
#         print(type(now_new))
#         now_new = now_new - datetime.timedelta(microseconds=now_new.microseconds)
#         # all_time = now_new.strftime('%H:%M:%S')
#         print(" сек. Всего прошло: " +
#                                   str(now_new))
#         sleep(0.1)

# -*- coding: utf-8 -*-
# import sys
# from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect
# from PyQt6.QtWidgets import (
#     QApplication, QMainWindow, QTextEdit, QListWidget,
#     QSplitter, QPushButton, QWidget, QVBoxLayout, QHBoxLayout
# )
#
# class SlidingPanel(QWidget):
#     """Виджет‑обёртка, который умеет плавно выезжать/съезжать."""
#     def __init__(self, widget, width=200, duration=250, parent=None):
#         super().__init__(parent)
#         self._content = widget
#         self._width = width
#         self._duration = duration
#
#         self.setFixedWidth(self._width)
#         self._content.setParent(self)
#
#         # Скрываем панель сразу (смещаем за границу)
#         self.hide()          # в начале не показываем
#         self._animation = QPropertyAnimation(self, b'geometry')
#         self._animation.setDuration(self._duration)
#         self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
#
#     def toggle(self):
#         if self.isVisible():
#             self._slide_out()
#         else:
#             self._slide_in()
#
#     def _slide_in(self):
#         self.show()
#         start = QRect(-self._width, 0, self._width, self.parent().height())
#         end   = QRect(0, 0, self._width, self.parent().height())
#         self._animation.setStartValue(start)
#         self._animation.setEndValue(end)
#         self._animation.start()
#
#     def _slide_out(self):
#         start = QRect(0, 0, self._width, self.parent().height())
#         end   = QRect(-self._width, 0, self._width, self.parent().height())
#         self._animation.setStartValue(start)
#         self._animation.setEndValue(end)
#         self._animation.start()
#         self._animation.finished.connect(self.hide)
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle('Выезжающая боковая панель')
#         self.resize(900, 600)
#
#         # ---------- Центральный редактор ----------
#         editor = QTextEdit()
#         editor.setPlainText('Тут основной контент...')
#
#         # ---------- Список, который будет в боковой панели ----------
#         side_list = QListWidget()
#         for i in range(1, 31):
#             side_list.addItem(f'Элемент {i}')
#
#         # ---------- Виджет‑обёртка с анимацией ----------
#         self.side_panel = SlidingPanel(side_list, width=250, duration=300, parent=self)
#
#         # ---------- Кнопка переключения ----------
#         toggle_btn = QPushButton('☰')
#         toggle_btn.setFixedSize(30, 30)
#         toggle_btn.clicked.connect(self.side_panel.toggle)
#
#         # ---------- Раскладка ----------
#         top_bar = QWidget()
#         top_layout = QHBoxLayout(top_bar)
#         top_layout.setContentsMargins(5, 5, 5, 5)
#         top_layout.addWidget(toggle_btn)
#         top_layout.addStretch()
#
#         central_widget = QWidget()
#         central_layout = QVBoxLayout(central_widget)
#         central_layout.setContentsMargins(0, 0, 0, 0)
#         central_layout.addWidget(top_bar)
#         central_layout.addWidget(editor)
#
#         # ---------- QSplitter (чтобы панель могла занимать место) ----------
#         splitter = QSplitter(Qt.Orientation.Horizontal)
#         splitter.addWidget(self.side_panel)   # слева будет панель (по умолчанию скрыта)
#         splitter.addWidget(central_widget)
#         splitter.setSizes([0, 1])  # панель стартует с нулевой шириной
#
#         self.setCentralWidget(splitter)

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     win = MainWindow()
#     win.show()
#     sys.exit(app.exec())

# print(int(''))
import torch
print(torch.__version__)
print(torch.cuda.is_available())
