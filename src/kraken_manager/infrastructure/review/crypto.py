"""Optional Ed25519 signatures and password-based AES-256-GCM envelope."""

from __future__ import annotations

import base64
import json
import os
import struct
from dataclasses import dataclass


MAGIC = b"KRAKEN-REVIEW\x01"


def _crypto() -> tuple[object, object, object, object]:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install Kraken with the 'packages' extra for signed/encrypted review packages") from exc
    return serialization, Ed25519PrivateKey, Ed25519PublicKey, AESGCM


@dataclass(frozen=True, slots=True)
class Ed25519KeyPair:
    private_key: bytes
    public_key: bytes

    @classmethod
    def generate(cls) -> "Ed25519KeyPair":
        serialization, private_type, _, _ = _crypto()
        private = private_type.generate()
        return cls(
            private.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            ),
            private.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            ),
        )


def sign(payload: bytes, private_key: bytes) -> str:
    _, private_type, _, _ = _crypto()
    return base64.b64encode(private_type.from_private_bytes(private_key).sign(payload)).decode("ascii")


def verify(payload: bytes, signature: str, public_key: bytes) -> bool:
    _, _, public_type, _ = _crypto()
    try:
        public_type.from_public_bytes(public_key).verify(base64.b64decode(signature, validate=True), payload)
    except Exception:
        return False
    return True


def encrypt_archive(payload: bytes, password: str) -> bytes:
    if len(password) < 12:
        raise ValueError("Package password must contain at least 12 characters")
    _, _, _, aes_type = _crypto()
    salt = os.urandom(16)
    nonce = os.urandom(12)
    key = __import__("hashlib").scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**15,
        r=8,
        p=1,
        dklen=32,
        maxmem=128 * 1024 * 1024,
    )
    header = json.dumps(
        {
            "cipher": "AES-256-GCM",
            "kdf": "scrypt-n32768-r8-p1",
            "salt": base64.b64encode(salt).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    ciphertext = aes_type(key).encrypt(nonce, payload, MAGIC + header)
    return MAGIC + struct.pack(">I", len(header)) + header + ciphertext


def decrypt_archive(envelope: bytes, password: str) -> bytes:
    if not envelope.startswith(MAGIC) or len(envelope) < len(MAGIC) + 4:
        raise ValueError("Not a Kraken review package")
    offset = len(MAGIC)
    header_size = struct.unpack(">I", envelope[offset : offset + 4])[0]
    if header_size < 1 or header_size > 16_384:
        raise ValueError("Invalid encrypted package header")
    offset += 4
    header_bytes = envelope[offset : offset + header_size]
    header = json.loads(header_bytes)
    if header.get("cipher") != "AES-256-GCM" or header.get("kdf") != "scrypt-n32768-r8-p1":
        raise ValueError("Unsupported package encryption")
    salt = base64.b64decode(header["salt"], validate=True)
    nonce = base64.b64decode(header["nonce"], validate=True)
    key = __import__("hashlib").scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**15,
        r=8,
        p=1,
        dklen=32,
        maxmem=128 * 1024 * 1024,
    )
    _, _, _, aes_type = _crypto()
    try:
        return aes_type(key).decrypt(nonce, envelope[offset + header_size :], MAGIC + header_bytes)
    except Exception as exc:
        raise ValueError("Invalid password or damaged encrypted package") from exc


__all__ = ["Ed25519KeyPair", "decrypt_archive", "encrypt_archive", "sign", "verify"]
