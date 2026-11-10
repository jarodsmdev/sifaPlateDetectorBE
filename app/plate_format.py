"""Normalización y validación de patentes chilenas.

Concentra el conocimiento de formatos vigentes (autos antiguos/nuevos, motos)
y del formato futuro de autos. También selecciona el mejor candidato que
entrega el OCR. No depende de OpenCV, PaddleOCR ni FastAPI.
"""

from __future__ import annotations

import os
import re
from typing import Iterable

# Formato nuevo de autos: 18 consonantes, sin vocales ni M, N, Q.
NEW_CAR_LETTERS = "BCDFGHJKLPRSTVWXYZ"
# El formato antiguo usó un alfabeto amplio; se mantiene permisivo.
OLD_CAR_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Motos: consonantes, incluyendo M, N y Q.
MOTO_LETTERS = "BCDFGHJKLMNPQRSTVWXYZ"

FORMAT_NEW_CAR = "nuevo"
FORMAT_OLD_CAR = "antiguo"
FORMAT_NEW_MOTO = "moto_nuevo"
FORMAT_OLD_MOTO = "moto_antiguo"
FORMAT_FUTURE_CAR = "futuro"

FORMATS = (
    (FORMAT_NEW_CAR, "LLLLDD", NEW_CAR_LETTERS),
    (FORMAT_OLD_CAR, "LLDDDD", OLD_CAR_LETTERS),
    (FORMAT_NEW_MOTO, "LLLDD", MOTO_LETTERS),
    (FORMAT_OLD_MOTO, "LLDDD", MOTO_LETTERS),
    (FORMAT_FUTURE_CAR, "LLLLLD", NEW_CAR_LETTERS),
)

_LETTER_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "E": "3",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}

_DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}

_DAY_PERIODS = {"dia", "noche"}
_ALLOWED_TAGS = {"suciedad", "angulo", "lluvia", "moto"}

# Más de dos correcciones ya no es una confusión visual razonable.
MAX_CORRECTIONS = 2


def _build_pattern(template: str, letters: str) -> re.Pattern:
    parts = [f"[{letters}]" if token == "L" else r"\d" for token in template]
    return re.compile("^" + "".join(parts) + "$")


_FORMAT_PATTERNS = {
    name: _build_pattern(template, letters)
    for name, template, letters in FORMATS
}


def normalize_plate(text: str) -> str:
    """Deja solo A-Z y 0-9 en mayúsculas."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def detect_plate_format(plate: str) -> str | None:
    """Devuelve el nombre del formato que calza exactamente, o None."""
    normalized = normalize_plate(plate)
    for name, _template, _letters in FORMATS:
        if _FORMAT_PATTERNS[name].match(normalized):
            return name
    return None


def _coerce_to_format(text: str, template: str, letters: str):
    """Fuerza cada posición al tipo esperado usando confusiones visuales."""
    if len(text) != len(template):
        return None

    allowed = set(letters)
    corrected = []
    changes = 0

    for char, token in zip(text, template):
        if token == "L":
            if char in allowed:
                new_char = char
            else:
                new_char = _DIGIT_TO_LETTER.get(char)
                if new_char is None or new_char not in allowed:
                    return None
        else:
            if char.isdigit():
                new_char = char
            else:
                new_char = _LETTER_TO_DIGIT.get(char)
                if new_char is None:
                    return None

        if new_char != char:
            changes += 1
        corrected.append(new_char)

    if changes > MAX_CORRECTIONS:
        return None

    return "".join(corrected), changes


def parse_ocr_candidates(candidates: Iterable[tuple[str, float]]) -> dict:
    """Elige el mejor texto OCR y lo valida contra los formatos chilenos.

    Prefiere un calce exacto, luego el que requiera menos correcciones y,
    finalmente, el de mayor confianza OCR.
    """
    best_valid = None
    best_invalid = None

    for raw_text, raw_score in candidates:
        text = normalize_plate(raw_text)
        if not text:
            continue

        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0

        plausible = (
            5 <= len(text) <= 7
            and any(char.isalpha() for char in text)
            and any(char.isdigit() for char in text)
        )
        if plausible and (best_invalid is None or score > best_invalid[0]):
            best_invalid = (score, text)

        for _name, template, letters in FORMATS:
            coerced = _coerce_to_format(text, template, letters)
            if coerced is None:
                continue
            corrected, changes = coerced
            key = (changes == 0, -changes, score)
            if best_valid is None or key > best_valid[0]:
                best_valid = (key, corrected, score)

    if best_valid is not None:
        _key, plate, score = best_valid
        return {
            "plate": plate,
            "plate_valid": True,
            "ocr_confidence": score,
            "status": "OK",
        }

    if best_invalid is not None:
        score, text = best_invalid
        return {
            "plate": text,
            "plate_valid": False,
            "ocr_confidence": score,
            "status": "Formato de patente no reconocido",
        }

    return {
        "plate": "",
        "plate_valid": False,
        "ocr_confidence": 0.0,
        "status": "La detección OCR no es confiable",
    }


def parse_validation_filename(filename: str) -> dict:
    """Interpreta ``dia_angulo_BBBB10.jpg`` y reporta etiquetas inválidas.

    El último token es la patente. El primero puede ser ``dia`` o ``noche``;
    los tokens intermedios son etiquetas libres del dataset.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = [part for part in base.split("_") if part]
    warnings = []

    if not parts:
        return {
            "plate": "",
            "day_period": "desconocido",
            "tags": [],
            "warnings": ["nombre vacío"],
        }

    plate = normalize_plate(parts[-1])
    tags = parts[:-1]
    day_period = "desconocido"

    if tags and tags[0].lower() in _DAY_PERIODS:
        day_period = tags[0].lower()
        tags = tags[1:]

    tags = [tag.lower() for tag in tags]

    if day_period == "desconocido":
        warnings.append("falta día/noche")
    for tag in tags:
        if tag not in _ALLOWED_TAGS:
            warnings.append(f"tag desconocido: {tag}")
    if detect_plate_format(plate) is None:
        warnings.append(f"patente no válida: {plate}")

    return {
        "plate": plate,
        "day_period": day_period,
        "tags": tags,
        "warnings": warnings,
    }
