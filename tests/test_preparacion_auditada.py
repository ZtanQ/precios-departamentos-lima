"""Comprobaciones sobre la fuente auditada, sin entrenar modelos."""
import unittest

import pandas as pd

from src.datos import cargar_datos
from src.preparacion import excluir_precios_auditados, preparar_base_actual


class PreparacionAuditadaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = cargar_datos()

    def test_exclusiones_y_conservaciones(self):
        original = self.raw.copy(deep=True)
        resultado = excluir_precios_auditados(self.raw)
        self.assertEqual(len(self.raw) - len(resultado), 17)
        self.assertNotIn(88630, resultado.index)
        for indice in [89031, 97233, 81729, 112895]:
            pd.testing.assert_series_equal(resultado.loc[indice], self.raw.loc[indice])
        for indice in [104708, 104776, 105020, 105044, 105244, 105270, 106243, 106303]:
            self.assertIn(indice, resultado.index)
        pd.testing.assert_frame_equal(original, self.raw)

    def test_fuente_modificada_rechazada(self):
        alterada = self.raw.copy()
        alterada.loc[88630, 'precio_usd'] = 190000
        with self.assertRaises(ValueError):
            excluir_precios_auditados(alterada)

    def test_base_y_faltantes(self):
        base = preparar_base_actual(self.raw)
        self.assertEqual(len(base), 40347)
        self.assertEqual(base.distrito.nunique(), 22)
        self.assertEqual(base.duplicated().sum(), 4)
        self.assertEqual(base.banos.isna().sum(), 1)
        self.assertEqual(base.antiguedad_anios.isna().sum(), 1)


if __name__ == '__main__':
    unittest.main()
