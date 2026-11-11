"""Pruebas unitarias de normalización, formatos y selección de candidatos."""
from app.plate_format import (
    FORMAT_FUTURE_CAR,
    FORMAT_NEW_CAR,
    FORMAT_NEW_MOTO,
    FORMAT_OLD_CAR,
    FORMAT_OLD_MOTO,
    detect_plate_format,
    normalize_plate,
    parse_ocr_candidates,
    parse_validation_filename,
)


def test_normalize_plate():
    assert normalize_plate(" ab-12 34 ") == "AB1234"
    assert normalize_plate("tzpw11.") == "TZPW11"
    assert normalize_plate(None) == ""


def test_detect_formatos_vigentes():
    assert detect_plate_format("BBBB10") == FORMAT_NEW_CAR
    assert detect_plate_format("TZPW11") == FORMAT_NEW_CAR
    assert detect_plate_format("AB1234") == FORMAT_OLD_CAR
    assert detect_plate_format("GZB74") == FORMAT_NEW_MOTO
    assert detect_plate_format("XG440") == FORMAT_OLD_MOTO
    assert detect_plate_format("BBBBB1") == FORMAT_FUTURE_CAR


def test_formato_nuevo_excluye_vocales_y_m_n_q():
    assert detect_plate_format("ABBB10") is None
    assert detect_plate_format("MBBB10") is None
    assert detect_plate_format("NBBB10") is None
    assert detect_plate_format("QBBB10") is None
    assert detect_plate_format("BBBB10") == FORMAT_NEW_CAR


def test_parse_prefiere_calce_exacto():
    resultado = parse_ocr_candidates([
        ("BBBB1O", 0.99),
        ("BBBB10", 0.80),
    ])
    assert resultado["plate"] == "BBBB10"
    assert resultado["plate_valid"] is True
    assert resultado["ocr_confidence"] == 0.80


def test_parse_corrige_confusiones_en_posicion_numerica():
    resultado = parse_ocr_candidates([("TZPW1I", 0.91)])
    assert resultado["plate"] == "TZPW11"
    assert resultado["plate_valid"] is True
    assert resultado["ocr_confidence"] == 0.91


def test_parse_ignora_texto_sin_digitos():
    resultado = parse_ocr_candidates([("CHILE", 0.99)])
    assert resultado["plate"] == ""
    assert resultado["plate_valid"] is False
    assert resultado["status"] == "La detección OCR no es confiable"


def test_parse_devuelve_mejor_esfuerzo_si_el_formato_no_calza():
    resultado = parse_ocr_candidates([("AB12345", 0.77)])
    assert resultado["plate"] == "AB12345"
    assert resultado["plate_valid"] is False
    assert resultado["status"] == "Formato de patente no reconocido"


def test_parse_sin_candidatos():
    resultado = parse_ocr_candidates([])
    assert resultado["plate"] == ""
    assert resultado["plate_valid"] is False
    assert resultado["ocr_confidence"] == 0.0


def test_parse_validation_filename_completo():
    parsed = parse_validation_filename("dia_angulo_BBBB10.jpg")
    assert parsed["plate"] == "BBBB10"
    assert parsed["day_period"] == "dia"
    assert parsed["tags"] == ["angulo"]
    assert parsed["warnings"] == []


def test_parse_validation_filename_sin_dia_noche():
    parsed = parse_validation_filename("angulo_BBBB10.jpg")
    assert parsed["day_period"] == "desconocido"
    assert parsed["tags"] == ["angulo"]
    assert "falta día/noche" in parsed["warnings"]


def test_parse_validation_filename_tag_desconocido():
    parsed = parse_validation_filename("dia_foo_BBBB10.jpg")
    assert "tag desconocido: foo" in parsed["warnings"]


def test_parse_validation_filename_patente_invalida():
    parsed = parse_validation_filename("dia_AB12345.jpg")
    assert "patente no válida: AB12345" in parsed["warnings"]
