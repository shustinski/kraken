"""Qt adapter for the remote job lifecycle."""

from __future__ import annotations

import json
from dataclasses import asdict
import threading
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError

from PyQt6.QtCore import QSettings, QThread, pyqtSignal

from .client import RemoteClient
from .contracts import TERMINAL, package_inputs


class RemoteJobThread(QThread):
    ask = pyqtSignal(str, str, bool, int)
    answer = pyqtSignal(bool)
    job_created = pyqtSignal(str)

    def __init__(self, *, url, main_state, settings_state, message_bus, job_id=None, request_key=None):
        super().__init__()
        preferences = QSettings()
        self.client_id = str(preferences.value("execution/client_id", "")) or uuid.uuid4().hex
        preferences.setValue("execution/client_id", self.client_id)
        self.url = url
        self.client = None
        self.main_state = main_state
        self.settings_state = settings_state
        self.bus = message_bus
        self.job_id = job_id
        self.request_key = request_key or uuid.uuid4().hex
        self.detached = threading.Event()
        self.cancel_upload = threading.Event()
        self.question_id = None
        self.remote_status = None
        self.pending_command = None
        self.pending_answer = None
        self.answer.connect(self._answer)

    def _answer(self, value):
        self.pending_answer = (self.question_id, bool(value))

    def stop(self):
        self.pending_command = "cancel"
        self.cancel_upload.set()

    def pause(self):
        self.pending_command = "pause"

    def detach(self):
        self.detached.set()
        self.cancel_upload.set()

    def run(self):
        try:
            self.client = RemoteClient(self.url, timeout=10)
            self.client.capabilities()
            destination = (
                Path(self.main_state.result_folder)
                if self.main_state.result_folder
                else Path(self.main_state.sample_folder).parent / "NeuralImageResults"
            ).resolve()
            destination.mkdir(parents=True, exist_ok=True)
            if self.job_id is None:
                payload, sources = package_inputs(self.main_state, self.settings_state, self.cancel_upload)
                payload["client_id"] = self.client_id
                record = destination / f".neuralimage-remote-{self.request_key}.json"
                recovery = {
                    "url": self.client.url.removesuffix("/api/v1"),
                    "job": None,
                    "request_key": self.request_key,
                    "destination": str(destination),
                    "main": asdict(self.main_state),
                    "settings": asdict(self.settings_state),
                }
                record.write_text(json.dumps(recovery), encoding="utf-8")
                created = self.client.request("POST", "/jobs", {"request_key": self.request_key, "payload": payload})
                self.job_id = created["id"]
                self.job_created.emit(self.job_id)
                recovery["job"] = self.job_id
                temporary = record.with_suffix(".tmp")
                temporary.write_text(json.dumps(recovery), encoding="utf-8")
                temporary.replace(record)
                self.bus.publish("logging", f"Remote job: {self.job_id}")
                self.client.upload(
                    self.job_id,
                    sources,
                    lambda current, total: self.bus.publish("logging", f"Upload: {current}/{total} bytes"),
                    cancelled=self.cancel_upload,
                )
                self.client.request("POST", f"/jobs/{self.job_id}/submit")
            else:
                state = self.client.request("GET", f"/jobs/{self.job_id}")
                if state["status"] == "uploading":
                    _, sources = package_inputs(self.main_state, self.settings_state, self.cancel_upload)
                    self.client.upload(self.job_id, sources, cancelled=self.cancel_upload)
                    self.client.request("POST", f"/jobs/{self.job_id}/submit")
                elif state["status"] in {"paused", "interrupted"}:
                    self.client.request("POST", f"/jobs/{self.job_id}/resume")
            cursor = 0
            while not self.detached.is_set():
                try:
                    if self.pending_command:
                        command = self.pending_command
                        self.client.request("POST", f"/jobs/{self.job_id}/{command}")
                        self.pending_command = None
                    if self.pending_answer:
                        question, value = self.pending_answer
                        self.client.request(
                            "POST", f"/jobs/{self.job_id}/answer", {"question": question, "value": value}
                        )
                        self.pending_answer = None
                    events = self.client.request("GET", f"/jobs/{self.job_id}/events?after={cursor}")
                    for event in events:
                        if event["topic"] == "question":
                            value = event["payload"]
                            self.question_id = value["id"]
                            self.ask.emit(value["text"], value["title"], value["default"], value["timeout"] or 0)
                        else:
                            self.bus.publish(event["topic"], self._decode(event["payload"]))
                        cursor = event["seq"]
                    state = self.client.request("GET", f"/jobs/{self.job_id}")
                    self.remote_status = state["status"]
                    if (state["status"] in TERMINAL or state["status"] == "paused") and not events:
                        if state["status"] == "succeeded":
                            try:
                                folder = self.client.download(self.job_id, destination, cancelled=self.detached)
                                self.bus.publish("logging", f"Results downloaded: {folder}")
                            except (OSError, ValueError) as error:
                                self.bus.publish(
                                    "logging",
                                    f"Computation succeeded; download must be retried for {self.job_id}: {error}",
                                )
                        elif state["status"] in {"failed", "interrupted"}:
                            self.bus.publish("error", state["error"])
                        break
                    if not events:
                        self.detached.wait(0.5)
                except HTTPError as error:
                    if error.code not in {502, 503, 504}:
                        raise
                    self.bus.publish("logging", f"Server unavailable; reconnecting: {error}")
                    self.detached.wait(3)
                except (URLError, TimeoutError) as error:
                    self.bus.publish("logging", f"Connection lost; reconnecting to job {self.job_id}: {error}")
                    self.detached.wait(3)
        except InterruptedError:
            if not self.detached.is_set() and self.job_id:
                try:
                    self.client.request("POST", f"/jobs/{self.job_id}/cancel")
                    self.remote_status = "cancelled"
                except (OSError, ValueError) as error:
                    self.bus.publish("error", f"Cannot cancel remote job {self.job_id}: {error}")
        except Exception as error:
            if not self.detached.is_set():
                self.bus.publish("error", str(error))

    def _decode(self, value):
        if isinstance(value, dict):
            if set(value) == {"array_artifact"}:
                import io
                import numpy as np
                from urllib.parse import quote
                from urllib.request import urlopen

                url = self.client.url + f"/jobs/{self.job_id}/artifacts/" + quote(value["array_artifact"], safe="/")
                with urlopen(url, timeout=self.client.timeout) as response:
                    return np.load(io.BytesIO(response.read()), allow_pickle=False)
            if set(value) == {"array", "dtype"}:
                import numpy as np

                return np.asarray(value["array"], dtype=value["dtype"])
            return {k: self._decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._decode(v) for v in value]
        return value
