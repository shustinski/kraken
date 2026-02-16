# glob = 1
#
# def plis():
#     global  glob
#     glob += 1
#     print("glob1 ", glob)
#     plus2()
#
#
# def plus2():
#     global  glob
#     print("glob2 ", glob)
#
# plis()
# print(1)
# plus2()

# import sys
# from PyQt6.QtWidgets import (
#     QMainWindow,
#     QWidget,
#     QLabel,
#     QLineEdit,
#     QPushButton,
#     QVBoxLayout,
#     QMessageBox, QApplication,
# )
# from PyQt6.QtCore import Qt
#
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self. setWindowTitle("Регистрация")
#         self.setGeometry(100, 100, 350, 250)
#
#         # CSS стили
#         style_sheet = """
#             /* Стили для главного окна */
#             QMainWindow {
#                 background-color: #f0f4f8;
#             }
#
#             /* Стили для центрального виджета */
#             QWidget {
#                 background-color: #ffffff;
#                 border-radius: 10px;
#                 padding: 20px;
#                 box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
#             }
#
#             /* Стили для меток */
#             QLabel {
#                 font-family: Arial, sans-serif;
#                 font-size: 14px;
#                 color: #333;
#                 margin-bottom: 5px;
#             }
#
#             /* Стили для полей ввода */
#             QLineEdit {
#                 font-family: Arial, sans-serif;
#                 font-size: 14px;
#                 padding: 8px;
#                 border: 2px solid #ddd;
#                 border-radius: 5px;
#                 background-color: #f8f9fa;
#                 margin-bottom: 15px;
#             }
#
#             QLineEdit:focus {
#                 border: 2px solid #4CAF50;
#                 outline: none;
#             }
#
#             /* Стили для кнопки */
#             QPushButton {
#                 font-family: Arial, sans-serif;
#                 font-size: 14px;
#                 background-color: #4CAF50;
#                 color: white;
#                 padding: 8px 20px;
#                 border: none;
#                 border-radius: 5px;
#                 cursor: pointer;
#             }
#
#             QPushButton:hover {
#                 background-color: #45a049;
#             }
#
#             QPushButton:pressed {
#                 background-color: #3d813f;
#             }
#         """
#
#         # Создаем виджеты
#         self.name_label = QLabel("Имя:")
#         self.name_input = QLineEdit()
#         self.surname_label = QLabel("Фамилия:")
#         self.surname_input = QLineEdit()
#         self.login_label = QLabel("Логин:")
#         self.login_input = QLineEdit()
#         self.submit_button = QPushButton("Сохранить")
#
#         # Создаем макет
#         layout = QVBoxLayout()
#         layout.addWidget(self.name_label)
#         layout.addWidget(self.name_input)
#         layout.addWidget(self.surname_label)
#         layout.addWidget(self.surname_input)
#         layout.addWidget(self.login_label)
#         layout.addWidget(self.login_input)
#         layout.addWidget(self.submit_button)
#
#         # Устанавливаем стиль для виджетов
#         self.setStyleSheet(style_sheet)
#         central_widget = QWidget()
#         central_widget.setLayout(layout)
#         central_widget.setContentsMargins(20, 20, 20, 20)
#         # central_widget.setAlignment(Qt.Alignment.Center)
#         self.setCentralWidget(central_widget)
#
#         # Подключаем обработчик для кнопки
#         self.submit_button.clicked.connect(self.on_submit)
#
#     def on_submit(self):
#         name = self.name_input.text()
#         surname = self.surname_input.text()
#         login = self.login_input.text()
#
#         if not name or not surname or not login:
#             QMessageBox.warning(
#                 self, "Ошибка", "Пожалуйста, заполните все поля."
#             )
#         else:
#             QMessageBox.information(
#                 self, "Успех", f"Данные сохранены:\nИмя: {name}\nФамилия: {surname}\nЛогин: {login}"
#             )
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec())
# import os
# print(os.environ['LOCAL_RANK'])

# demo_spinbox.py
import sys
from PyQt6 import QtWidgets

STYLE = """
QSpinBox,
QDoubleSpinBox {
    background-color: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding-left: 4px;
    padding-right: 30px;
    min-height: 34px;
}
QSpinBox::up-button,
QDoubleSpinBox::up-button,
QSpinBox::down-button,
QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 24px;
    height: 16px;
    background: #f5f5f5;
    border-left: 1px solid #e0e0e0;
}
QSpinBox::down-button,
QDoubleSpinBox::down-button {
    subcontrol-position: bottom right;
}
QSpinBox::up-arrow,
QDoubleSpinBox::up-arrow {
    content: "▲";
    font-size: 9pt;
    color: #555;
    margin: 0;
}
QSpinBox::down-arrow,
QDoubleSpinBox::down-arrow {
    content: "▼";
    font-size: 9pt;
    color: #555;
    margin: 0;
}
QSpinBox::up-button:hover,
QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover,
QDoubleSpinBox::down-button:hover {
    background: #7ea6ff;
}
QSpinBox::up-button:pressed,
QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed,
QDoubleSpinBox::down-button:pressed {
    background: #2a5bd9;
}
"""

def main():
    # On macOS the native style hides the arrows – force Fusion.
    QtWidgets.QApplication.setStyle("Fusion")

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLE)

    # ------------------------------------------------------------
    # Build a tiny UI that contains a QSpinBox and a QDoubleSpinBox
    # ------------------------------------------------------------
    win = QtWidgets.QWidget()
    win.setWindowTitle("SpinBox demo – arrows should be visible")
    layout = QtWidgets.QVBoxLayout(win)

    sb = QtWidgets.QSpinBox()
    sb.setRange(0, 100)
    sb.setValue(42)
    layout.addWidget(sb)

    dsb = QtWidgets.QDoubleSpinBox()
    dsb.setRange(-5.0, 5.0)
    dsb.setDecimals(2)
    dsb.setValue(1.23)
    layout.addWidget(dsb)

    win.resize(200, 120)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()