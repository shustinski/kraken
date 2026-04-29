from __future__ import annotations

ASCII_SYMBOLS = (
    "`1234567890-=\t"
    "qwertyuiop[]\\asdfghjkl;'zxcvbnm,./"
    'QWERTYUIOP{}|ASDFGHJKL:"ZXCVBNM<>? _+'
)
CYRILLIC_SYMBOLS = "абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
ALPHABET = tuple(dict.fromkeys(ASCII_SYMBOLS + CYRILLIC_SYMBOLS))
_INDEX = {symbol: index for index, symbol in enumerate(ALPHABET)}


class CipherError(ValueError):
    pass


def validate_decodable(text: object) -> tuple[bool, str | None]:
    for char in str(text):
        if char not in _INDEX:
            return False, char
    return True, None


def _shift_text(text: str, key: str, direction: int) -> str:
    if not key:
        raise CipherError("Encryption key must not be empty.")
    ok, invalid = validate_decodable(text)
    if not ok:
        raise CipherError(f"Text contains unsupported character: {invalid!r}.")
    ok, invalid = validate_decodable(key)
    if not ok:
        raise CipherError(f"Key contains unsupported character: {invalid!r}.")
    alphabet_len = len(ALPHABET)
    result: list[str] = []
    for index, char in enumerate(text):
        key_char = key[index % len(key)]
        result.append(ALPHABET[(_INDEX[char] + direction * _INDEX[key_char]) % alphabet_len])
    return "".join(result)


def encode_string(plain_text: object, key: str) -> str:
    return _shift_text(str(plain_text), key, 1)


def decode_string(encoded_string: str, key: str) -> str:
    return _shift_text(encoded_string, key, -1)
