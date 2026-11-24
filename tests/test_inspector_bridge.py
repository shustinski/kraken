"""The development bridge must drive actual Qt events through a local socket."""

from __future__ import annotations

import json
from uuid import uuid4

from PyQt6.QtNetwork import QLocalSocket
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLineEdit, QMainWindow, QPushButton, QWidget

from kraken_inspector.bridge import InspectorBridge


def _request(app: QApplication, name: str, payload: dict) -> dict:
    socket = QLocalSocket()
    socket.connectToServer(name)
    assert socket.waitForConnected(1000)
    socket.write(json.dumps(payload).encode() + b"\n")
    socket.flush()
    for _ in range(200):
        app.processEvents()
        if socket.canReadLine():
            return json.loads(bytes(socket.readLine()))
        QTest.qWait(10)
    raise AssertionError("bridge did not respond")


def test_bridge_discovers_and_clicks_a_real_button(qtbot) -> None:
    app = QApplication.instance()
    window = QMainWindow()
    button = QPushButton("Create", window)
    button.setObjectName("createButton")
    window.setCentralWidget(button)
    qtbot.addWidget(window)
    window.show()
    clicks = []
    button.clicked.connect(lambda: clicks.append(True))
    name = f"kraken-inspector-test-{uuid4().hex}"
    bridge = InspectorBridge(name, "secret")
    try:
        denied = _request(app, name, {"token": "wrong", "op": "inspect"})
        assert denied["error"] == "unauthorized"
        result = _request(app, name, {"token": "secret", "op": "inspect"})
        nodes = result["windows"][0]["nodes"]
        target = next(node for node in nodes if node["object_name"] == "createButton")
        assert target["text"] == "Create"
        assert target["enabled"] and target["visible"]
        action = _request(app, name, {"token": "secret", "op": "action", "id": target["id"], "action": "click"})
        assert action["ok"]
        assert clicks == [True]
        stale = _request(app, name, {"token": "secret", "op": "action", "id": "w999:0", "action": "click"})
        assert stale["error"] == "element_not_found"
    finally:
        bridge.close()


def test_bridge_requires_unique_endpoint(qtbot) -> None:
    name = f"kraken-inspector-test-{uuid4().hex}"
    bridge = InspectorBridge(name, "secret")
    try:
        try:
            InspectorBridge(name, "other")
        except RuntimeError:
            pass
        else:
            raise AssertionError("second bridge stole first endpoint")
    finally:
        bridge.close()


def test_menu_action_uses_visible_menu_path(qtbot) -> None:
    app = QApplication.instance()
    window = QMainWindow()
    menu = window.menuBar().addMenu("File")
    selected = []
    menu.addAction("New", lambda: selected.append(True))
    qtbot.addWidget(window)
    window.show()
    name = f"kraken-inspector-test-{uuid4().hex}"
    bridge = InspectorBridge(name, "secret")
    try:
        nodes = _request(app, name, {"token": "secret", "op": "inspect"})["windows"][0]["nodes"]
        root = next(node for node in nodes if node["class"] == "QAction" and node["text"] == "File")
        child = next(node for node in nodes if node["class"] == "QAction" and node["text"] == "New")
        assert root["visible"]
        assert not child["visible"]
        assert (
            _request(app, name, {"token": "secret", "op": "action", "id": child["id"], "action": "click"})["error"]
            == "element_unavailable"
        )
        assert _request(app, name, {"token": "secret", "op": "action", "id": root["id"], "action": "click"})["ok"]
        app.processEvents()
        nodes = _request(app, name, {"token": "secret", "op": "inspect"})["windows"][0]["nodes"]
        child = next(node for node in nodes if node["class"] == "QAction" and node["text"] == "New")
        assert child["visible"]
        assert _request(app, name, {"token": "secret", "op": "action", "id": child["id"], "action": "click"})["ok"]
        app.processEvents()
        assert selected == [True]
    finally:
        bridge.close()


def test_manual_click_is_recorded_without_inspector_click(qtbot) -> None:
    app = QApplication.instance()
    window = QMainWindow()
    button = QPushButton("Save", window)
    window.setCentralWidget(button)
    qtbot.addWidget(window)
    window.show()
    name = f"kraken-inspector-test-{uuid4().hex}"
    bridge = InspectorBridge(name, "secret")
    try:
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        result = _request(app, name, {"token": "secret", "op": "events"})
        assert any(event["action"] == "click" and event["text"] == "Save" for event in result["events"])
        button_id = next(
            node["id"]
            for node in _request(app, name, {"token": "secret", "op": "inspect"})["windows"][0]["nodes"]
            if node["text"] == "Save"
        )
        _request(app, name, {"token": "secret", "op": "action", "id": button_id, "action": "click"})
        app.processEvents()
        assert _request(app, name, {"token": "secret", "op": "events"})["events"] == []
    finally:
        bridge.close()


def test_password_field_is_not_exposed_or_recorded(qtbot) -> None:
    app = QApplication.instance()
    window = QMainWindow()
    password = QLineEdit(window)
    password.setEchoMode(QLineEdit.EchoMode.Password)
    password.setText("private-value")
    window.setCentralWidget(password)
    qtbot.addWidget(window)
    window.show()
    name = f"kraken-inspector-test-{uuid4().hex}"
    bridge = InspectorBridge(name, "secret")
    try:
        nodes = _request(app, name, {"token": "secret", "op": "inspect"})["windows"][0]["nodes"]
        field = next(node for node in nodes if node["class"] == "QLineEdit")
        assert field["text"] == "[redacted]" and field["sensitive"]
        assert (
            _request(
                app,
                name,
                {"token": "secret", "op": "action", "id": field["id"], "action": "type", "value": "another-secret"},
            )["error"]
            == "sensitive_field"
        )
        QTest.keyClicks(password, "x")
        assert _request(app, name, {"token": "secret", "op": "events"})["events"] == []
    finally:
        bridge.close()


def test_custom_widget_accepts_point_click(qtbot) -> None:
    app = QApplication.instance()
    points = []

    class Canvas(QWidget):
        def mouseReleaseEvent(self, event) -> None:
            points.append(event.position().toPoint())

    window = QMainWindow()
    canvas = Canvas()
    canvas.resize(200, 100)
    window.setCentralWidget(canvas)
    window.resize(200, 100)
    qtbot.addWidget(window)
    window.show()
    name = f"kraken-inspector-test-{uuid4().hex}"
    bridge = InspectorBridge(name, "secret")
    try:
        nodes = _request(app, name, {"token": "secret", "op": "inspect"})["windows"][0]["nodes"]
        target = next(node for node in nodes if node["class"] == "Canvas")
        result = _request(
            app, name, {"token": "secret", "op": "action", "id": target["id"], "action": "click", "x": 0.25, "y": 0.75}
        )
        assert result["ok"]
        app.processEvents()
        assert len(points) == 1
        assert abs(points[0].x() / canvas.width() - 0.25) < 0.05
        assert abs(points[0].y() / canvas.height() - 0.75) < 0.05
    finally:
        bridge.close()
