"""Genera la única base preparada actual, sin entrenar modelos ni cambiar cortes."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import config
from src.datos import cargar_datos
from src.preparacion import preparar_base_actual


def main():
    base = preparar_base_actual(cargar_datos())
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    destino = config.DATA_PROCESSED / "base_modelado.parquet"
    base.to_parquet(destino, index=False)
    resumen = base.groupby("anio").agg(anuncios=("precio_usd", "size"), distritos=("distrito", "nunique"))
    resumen.to_csv(config.RAIZ / "reports" / "auditoria_preparacion" / "conteos_base_actual.csv")
    print(f"Base actual: {destino}\nFilas: {len(base):,}\nDistritos: {base.distrito.nunique()}")
    print(resumen.to_string())
    print("Base previa al filtro IQR; imputación dentro del pipeline. No se entrenaron modelos.")


if __name__ == "__main__":
    main()
