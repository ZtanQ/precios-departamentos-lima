import unittest
import pandas as pd

from src.validacion import dividir_temporal, construir_entrega, separar_xy


def ejemplo():
    filas = []
    for anio, trimestre in [(2022, 1), (2023, 2), (2023, 3), (2023, 4), (2024, 1), (2026, 1)]:
        for i in range(12):
            filas.append(dict(fila_origen=len(filas), anio=anio, trimestre=trimestre,
                              distrito='Surco', superficie_m2=80., precio_usd=100000.+i*100,
                              habitaciones=3., banos=2., garajes=1., antiguedad_anios=10.))
    return pd.DataFrame(filas)


class ValidacionTemporalTest(unittest.TestCase):
    def test_fronteras_y_sin_solapamiento(self):
        base = ejemplo()
        partes = dividir_temporal(base)
        self.assertEqual(len(partes['entrenamiento']), 24)
        self.assertEqual(len(partes['validacion']), 24)
        self.assertEqual(len(partes['prueba']), 24)
        ids = [set(p.fila_origen) for p in partes.values()]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

    def test_test_no_altera_filtro_seleccion(self):
        base = ejemplo()
        a = construir_entrega(base)
        base.loc[base.anio.ge(2024), 'precio_usd'] = 1e10
        b = construir_entrega(base)
        pd.testing.assert_series_equal(a.incluido_iqr_seleccion, b.incluido_iqr_seleccion)
        self.assertTrue(b.loc[b.particion.eq('prueba'), 'incluido_iqr_seleccion'].isna().all())
        self.assertFalse(b.loc[b.particion.eq('prueba'), 'incluido_iqr_final'].any())
        X, y = separar_xy(b)
        self.assertNotIn('fila_origen', X)
        self.assertNotIn('incluido_iqr_final', X)
        self.assertNotIn('precio_usd', X)

    def test_cambio_validacion_no_cambia_mascara_train(self):
        base = ejemplo()
        a = construir_entrega(base)
        base.loc[base.anio.eq(2023) & base.trimestre.ge(3), 'precio_usd'] = 1e10
        b = construir_entrega(base)
        train = a.particion.eq('entrenamiento')
        pd.testing.assert_series_equal(a.loc[train, 'incluido_iqr_seleccion'], b.loc[train, 'incluido_iqr_seleccion'])

    def test_id_repetido_rechazado(self):
        base = ejemplo()
        base.loc[1,'fila_origen'] = 0
        with self.assertRaises(ValueError): dividir_temporal(base)


if __name__ == '__main__':
    unittest.main()
