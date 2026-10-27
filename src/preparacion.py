"""Exclusiones manuales acordadas tras la auditoría; no imputa el objetivo."""
import hashlib
import json

import pandas as pd

from src import config, limpieza

MANIFIESTO = config.RAIZ / "reports" / "auditoria_preparacion" / "decisiones_precios.csv"


def huella_fila(fila):
    """Firma de valores y columnas para impedir excluir otra fila por posición."""
    texto = json.dumps(list(zip(fila.index, fila.tolist())), ensure_ascii=False,
                       allow_nan=True, default=lambda valor: valor.item())
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def excluir_precios_auditados(df):
    """Recibe la carga ORIGINAL antes de ordenar, deduplicar o añadir columnas.

    Verifica posiciones y valores frente al manifiesto. Falla si la fuente cambió;
    requiere una nueva auditoría en vez de excluir silenciosamente otros anuncios.
    Devuelve una copia sin las 17 filas acordadas. El manifiesto incluye cuatro
    precios bajos conservados para registrar también esa decisión.
    """
    decisiones = pd.read_csv(MANIFIESTO)
    if len(df) != 119980 or list(df.columns) != list(config.NOMBRES_COLUMNAS.values()):
        raise ValueError("Se requiere cargar_datos() sin transformaciones y la fuente auditada de 119980 filas.")
    if decisiones.fila_origen.duplicated().any():
        raise ValueError("El manifiesto contiene posiciones repetidas.")
    for decision in decisiones.itertuples():
        if not 0 <= decision.fila_origen < len(df) or huella_fila(df.iloc[decision.fila_origen]) != decision.huella:
            raise ValueError(f"La fila {decision.fila_origen} cambió: revisar la fuente y repetir la auditoría.")
    posiciones = decisiones.loc[decisiones.decision.eq("excluir"), "fila_origen"]
    mascara = pd.Series(True, index=range(len(df)))
    mascara.iloc[posiciones] = False
    return df.iloc[mascara.to_numpy()].copy()


def preparar_base_actual(df, conservar_id=False):
    """Una base de trabajo previa a IQR; conserva la metodología restante del TP1.

    No imputa, escala ni codifica. Las coincidencias que difieren en piso se
    conservan porque la deduplicación ocurre sobre todas las columnas originales.
    """
    base = excluir_precios_auditados(df)
    # La trazabilidad no participa en la igualdad de las columnas originales.
    base = base.loc[~base.duplicated()].copy()
    if conservar_id:
        base["fila_origen"] = base.index.to_numpy()
    base = base.reset_index(drop=True)
    base = limpieza.normalizar_distritos(base)
    base = limpieza.corregir_valores_imposibles(base)
    descartar = config.COLUMNAS_FUGA + config.COLUMNAS_REDUNDANTES + ["piso", "vista_exterior"]
    base = base.drop(columns=descartar)
    # Se preserva el soporte global del TP1; su revisión pertenece al protocolo pendiente.
    base = limpieza.descartar_distritos_baja_frecuencia(base, min_anuncios=10)
    periodo = base.anio.between(2022, 2025) | (base.anio.eq(2026) & base.trimestre.eq(1))
    return base.loc[periodo].reset_index(drop=True)
