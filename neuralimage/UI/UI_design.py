from PyQt6.QtGui import QFont


class GuiStyle:

    button_style = "QPushButton {background-color:rgb(204, 204, 204); " \
                   "border-style: outset; border-width: 2px; " \
                   "border-radius: 10px; border-color: beige; padding: 6px;}" \
                   "QPushButton:hover {background-color: rgb(204, 239, 255);}"

    copy_button_style_disabled = "QPushButton {background-color:rgb(217,217,217); " \
                                 "border-style: outset; border-width: 2px; " \
                                 "border-radius: 10px; border-color: beige; padding: 6px;}" \
                                 "QPushButton:hover {background-color: rgb(204, 239, 255);}"
    copy_button_style_enabled = "QPushButton {background-color:rgb(153,255,204); " \
                                "border-style: outset; border-width: 2px; " \
                                "border-radius: 10px; border-color: beige; padding: 6px;}" \
                                "QPushButton:hover {background-color: rgb(102, 255, 102);}"

    finish_button_style = "QPushButton {background-color:rgb(255, 102, 102); " \
                          "border-style: outset; border-width: 2px; " \
                          "border-radius: 10px; border-color: beige; padding: 6px;}" \
                          "QPushButton:hover {background-color: rgb(255, 51, 51);}"

    path_label_style = "background-color: rgb(255,255,255); border-style: solid; border-width: 2px; border-color: gray;"

    insufficient_samples_style = """border-style: solid; border-width: 2px; border-color: red;
                                QToolTip {
                                background_color: white;
                                color: white
                                }"""
    sufficient_samples_style = "border-style: solid; border-width: 2px; border-color: green;"
    unknown_samples_style = "border-style: solid; border-width: 2px; border-color: grey;"

    style_sheet = """ 
               QMainWindow{background-color:#f0f4f8;}
            QWidget{
                background-color:#ffffff;
                border-radius:10px;
                padding:7px;
                box-shadow:0 2px 10px rgba(0,0,0,0.1);
            }
            QLabel{
                font-family:Arial,sans-serif;
                font-size:12px;
                color:#333;
                margin-bottom:5px;
            }
            QLineEdit{
                font-family:Arial,sans-serif;
                font-size:12px;
                padding:2px;
                border:2px solid #ddd;
                border-radius:5px;
                background-color:#f8f9fa;
                margin-bottom:2px;
            }
            ClickableLabel{
                font-family:Arial,sans-serif;
                font-size:12px;
                padding:8px;
                border:2px solid #ddd;
                border-radius:5px;
                background-color:#f8f9fa;
                margin-bottom:5px;
            }
            QLineEdit:focus{border:2px solid #4CAF50;outline:none;}
            QPushButton{
                font-family:Arial,sans-serif;
                font-size:12px;
                background-color:#4CAF50;
                color:white;
                padding:8px 20px;
                border:none;
                border-radius:5px;
                cursor:pointer;
            }
            QPushButton:hover{background-color:#45a049;}
            QPushButton:pressed{background-color:#3d813f;}
            /* Style for normal QAction items */
            QAction {
                color: black;
            }
            /* Style for hovered QAction items */
            QAction:hover {
                background-color: #4CAF50;
                color: black !important; /* Ensure text remains visible on hover */
            }
            /* Style for selected/active QAction items */
            
            /* Optional styling for the QMenu container */
            QMenu {
                border: 1px solid #888;
                padding: 2px;
                border-radius:5px;
                color: black;
            }
            QMenu:active,
            QMenu:selected {
                background-color: #45a049;
                color: black;
            }
            """

    font = QFont()
    font.setPointSize(10)
    font.setWeight(600)


    """
                """