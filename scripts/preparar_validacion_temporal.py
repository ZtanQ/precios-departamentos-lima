"""Exporta una base con particiones y evidencia del protocolo; no entrena modelos."""
import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import config
from src.datos import cargar_datos
from src.preparacion import preparar_base_actual, MANIFIESTO
from src.validacion import construir_entrega


def main():
    entrega = construir_entrega(preparar_base_actual(cargar_datos(), conservar_id=True))
    carpeta = RAIZ / "reports" / "validacion_temporal"
    carpeta.mkdir(parents=True, exist_ok=True)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    # Única base exportada, enriquecida con trazabilidad y particiones.
    entrega.to_parquet(config.DATA_PROCESSED / "base_modelado.parquet", index=False)
    resumen = entrega.groupby("particion", sort=False).agg(
        anuncios=("fila_origen", "size"), distritos=("distrito", "nunique"),
        anio_min=("anio", "min"), anio_max=("anio", "max"),
        incluidos_iqr_seleccion=("incluido_iqr_seleccion", lambda x: x.sum(min_count=1)),
        incluidos_iqr_final=("incluido_iqr_final", "sum"),
        banos_faltantes=("banos", lambda x: x.isna().sum()),
        antiguedad_faltante=("antiguedad_anios", lambda x: x.isna().sum()),
    )
    resumen.to_csv(carpeta / "resumen_particiones.csv", encoding="utf-8-sig")
    entrega.groupby(["particion", "anio", "trimestre"]).size().rename("anuncios").to_csv(carpeta / "cobertura_trimestral.csv", encoding="utf-8-sig")
    entrega.groupby(["distrito", "particion"]).size().unstack(fill_value=0).to_csv(carpeta / "cobertura_distrital.csv", encoding="utf-8-sig")
    soporte = entrega.loc[entrega.n_distrito_train_seleccion.lt(10), ["fila_origen", "distrito", "particion", "n_distrito_train_seleccion"]]
    soporte.to_csv(carpeta / "soporte_insuficiente.csv", index=False, encoding="utf-8-sig")
    resumen_texto = (
        "Validación temporal: ejecución local\n"
        "Entrenamiento: 2022-T1–2023-T2. Validación: 2023-T3/T4. Prueba: 2024-T1–2026-T1.\n"
        "No se entrenaron modelos. No se imputaron objetivos ni características.\n\n"
        + resumen.to_string() + "\n\n"
        "IQR selección ajustado solo en entrenamiento; máscara de prueba ausente intencionalmente.\n"
        "IQR final ajustado en 2022–2023 para reentrenamiento posterior a la selección.\n"
        "Las máscaras usan el precio observado: delimitan una muestra condicionada, no un filtro de inferencia.\n"
        "Los IDs y máscaras se excluyen con src.validacion.separar_xy().\n"
        "La prueba y exclusiones manuales ya se exploraron: no es una prueba inédita.\n"
        "El protocolo conserva normalización y soporte global del TP1; no demuestra ausencia absoluta de fuga.\n"
    )
    (carpeta / "resumen_validacion.txt").write_text(resumen_texto, encoding="utf-8")
    fuente = config.ARCHIVO_PARQUET if config.ARCHIVO_PARQUET.exists() else config.ARCHIVO_EXCEL
    metadata = {"python": sys.version, "fuente_sha256": hashlib.sha256(fuente.read_bytes()).hexdigest(),
                "decisiones_sha256": hashlib.sha256(MANIFIESTO.read_bytes()).hexdigest(),
                "filas": len(entrega), "modelos_entrenados": False}
    (carpeta / "ejecucion.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(resumen_texto)
    print(f"Evidencia guardada en: {carpeta}")


if __name__ == "__main__":
    main()
