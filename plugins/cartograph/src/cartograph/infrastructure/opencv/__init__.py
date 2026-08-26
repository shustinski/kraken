"""OpenCV-backed compute isolated from domain and application layers."""

from .registrar import PythonPairRegistrar, PythonRegistrationBackend

__all__ = ["PythonPairRegistrar", "PythonRegistrationBackend"]
