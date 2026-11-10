# Dataset local de validación

Las imágenes reales de patentes **no se versionan** (ver `.gitignore`). Solo se
versiona este README con la convención de nombres.

## Convención

```text
<dia|noche>_[etiqueta...]_<PATENTE>.<jpg|jpeg|png>
```

- El **último** token antes de la extensión es la patente real (Ground Truth).
- El **primero** puede ser `dia` o `noche`. Si falta, la evaluación lo reporta
  como `desconocido`.
- Los tokens intermedios son etiquetas libres del dataset.

Etiquetas usadas actualmente:

- `angulo`: patente inclinada o en perspectiva.
- `suciedad`: patente sucia o parcialmente tapada.
- `lluvia`: captura con lluvia.
- `moto`: motocicleta (formatos de 5 caracteres).

Ejemplos:

```text
dia_BBBB10.jpg
dia_angulo_BKJY33.jpg
noche_suciedad_KPZX17.jpg
dia_moto_GZB74.jpg
noche_angulo_ZK8624.jpg
```

## Evaluación

```bash
python test/evaluate_validation.py
# o el alias histórico:
python test/test_precision.py
```

Para exigir que todos los nombres cumplan la convención:

```bash
python test/evaluate_validation.py --strict-tags
```

## Formatos chilenos reconocidos

| Formato | Patrón | Ejemplo |
|---|---|---|
| Auto nuevo | 4 letras + 2 números | `TZPW11` |
| Auto antiguo | 2 letras + 4 números | `AB1234` |
| Moto nueva | 3 letras + 2 números | `GZB74` |
| Moto antigua | 2 letras + 3 números | `XG440` |
| Auto futuro | 5 letras + 1 número | `BBBBB1` |

El formato nuevo de autos usa 18 consonantes: excluye vocales, `M`, `N` y `Q`.
