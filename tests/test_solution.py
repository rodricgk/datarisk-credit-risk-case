import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import solution  # noqa: E402


class TargetAndMetricTests(unittest.TestCase):
    def test_target_boundary_is_five_days(self):
        due = pd.Timestamp("2021-01-10")
        frame = pd.DataFrame(
            {
                "DATA_VENCIMENTO": [due, due],
                "DATA_PAGAMENTO": [due + pd.Timedelta(days=4), due + pd.Timedelta(days=5)],
            }
        )
        result = solution.build_target(frame)
        self.assertEqual(result["INADIMPLENTE"].tolist(), [0, 1])

    def test_ks_is_invariant_to_order_inside_score_ties(self):
        y = pd.Series([0, 1, 0, 1])
        scores = np.array([0.9, 0.9, 0.1, 0.1])
        first = solution.ks_statistic(y, scores)
        second = solution.ks_statistic(y.iloc[[1, 0, 3, 2]], scores)
        self.assertAlmostEqual(first, second)

    def test_payment_before_emission_is_removed(self):
        frame = pd.DataFrame(
            {
                "DATA_EMISSAO_DOCUMENTO": [pd.Timestamp("2021-01-10"), pd.Timestamp("2021-01-10")],
                "DATA_VENCIMENTO": [pd.Timestamp("2021-01-20"), pd.Timestamp("2021-01-20")],
                "DATA_PAGAMENTO": [pd.Timestamp("2021-01-09"), pd.Timestamp("2021-01-21")],
            }
        )
        result = solution.filter_datas_inconsistentes(frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["DATA_PAGAMENTO"], pd.Timestamp("2021-01-21"))


class TemporalSplitTests(unittest.TestCase):
    def test_future_month_never_enters_training(self):
        frame = pd.DataFrame({"SAFRA_REF": ["2021-01", "2021-03", "2021-07"]})
        with self.assertRaisesRegex(ValueError, "posteriores"):
            solution.temporal_split(frame, ["2021-03"])


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.cadastral = pd.DataFrame(
            {
                "ID_CLIENTE": [1],
                "DATA_CADASTRO": ["2020-01-01"],
                "DDD": [11],
                "FLAG_PF": [np.nan],
                "SEGMENTO_INDUSTRIAL": ["SERVICOS"],
                "DOMINIO_EMAIL": ["gmail.com"],
                "PORTE": ["PEQUENO"],
                "CEP_2_DIG": [1],
            }
        )
        self.info = pd.DataFrame(
            {"ID_CLIENTE": [1], "SAFRA_REF": ["2021-01"], "RENDA_MES_ANTERIOR": [1000], "NO_FUNCIONARIOS": [2]}
        )
        self.dev = pd.DataFrame(
            {
                "ID_CLIENTE": [1],
                "SAFRA_REF": ["2021-01"],
                "DATA_EMISSAO_DOCUMENTO": [pd.Timestamp("2021-01-01")],
                "DATA_PAGAMENTO": [pd.Timestamp("2021-01-10")],
                "DATA_VENCIMENTO": [pd.Timestamp("2021-01-05")],
                "VALOR_A_PAGAR": [100.0],
                "TAXA": [0.1],
            }
        )
        self.test = self.dev.drop(columns="DATA_PAGAMENTO")

    def test_duplicate_dimension_key_fails_early(self):
        duplicated = pd.concat([self.info, self.info], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "deve ser unico"):
            solution.validate_input_data(self.cadastral, duplicated, self.dev, self.test)

    def test_profile_join_preserves_rows_and_marks_invalid_ddd(self):
        cadastral = self.cadastral.assign(DDD="(1")
        featured = solution.add_profile_features(self.dev, cadastral, self.info)
        self.assertEqual(len(featured), len(self.dev))
        self.assertEqual(featured.loc[0, "DDD"], "DESCONHECIDO")
        self.assertEqual(featured.loc[0, "DDD_INVALIDO"], 1)


if __name__ == "__main__":
    unittest.main()

