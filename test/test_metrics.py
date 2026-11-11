"""Pruebas unitarias de las métricas de evaluación."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_validation import calcular_mediana, calcular_moda, levenshtein_distance


def test_levenshtein_distancia():
    assert levenshtein_distance("BBBB10", "BBBB11") == 1
    assert levenshtein_distance("BBBB10", "BBBB10") == 0
    assert levenshtein_distance("", "AB1234") == 6


def test_mediana():
    assert calcular_mediana([]) == 0.0
    assert calcular_mediana([3]) == 3
    assert calcular_mediana([1, 3, 2]) == 2
    assert calcular_mediana([1, 2, 3, 4]) == 2.5


def test_moda():
    assert calcular_moda([]) == 0.0
    assert calcular_moda([1.11, 1.12, 1.11]) == 1.11
    assert calcular_moda([2.0, 2.0, 3.0]) == 2.0
