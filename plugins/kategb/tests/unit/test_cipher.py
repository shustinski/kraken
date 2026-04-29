from __future__ import annotations

import pytest

from kategb.domain.cipher import CipherError, decode_string, encode_string, validate_decodable


def test_cipher_roundtrip_preserves_legacy_manifest_text() -> None:
    text = r"C:\vectors\layer_1"
    key = "key_123"

    encoded = encode_string(text, key)

    assert encoded != text
    assert decode_string(encoded, key) == text


def test_cipher_rejects_unsupported_characters() -> None:
    ok, invalid = validate_decodable("bad🙂")

    assert not ok
    assert invalid == "🙂"
    with pytest.raises(CipherError):
        encode_string("bad🙂", "key")
