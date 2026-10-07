"""Regressoes dos achados da revisao; dados sinteticos, sem acesso a rede."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import solution as s


def payments():
    return pd.DataFrame({
        "ID_CLIENTE": [1, 1, 1, 2],
        "SAFRA_REF": ["2021-01", "2021-01", "2021-02", "2021-02"],
        "DATA_EMISSAO_DOCUMENTO": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-02-01", "2021-02-02"]),
        "DATA_VENCIMENTO": pd.to_datetime(["2021-01-10", "2021-01-30", "2021-02-10", "2021-03-02"]),
        "DATA_PAGAMENTO": pd.to_datetime(["2021-01-12", "2021-03-10", "2021-02-16", "2021-03-04"]),
        "VALOR_A_PAGAR": [100., np.nan, 20., 30.], "TAXA": [.1] * 4,
    })


def dimensions():
    c = pd.DataFrame({
        "ID_CLIENTE": [1, 2], "DATA_CADASTRO": ["2020-01-01"] * 2,
        "DDD": [11, 11], "FLAG_PF": [np.nan, "X"], "SEGMENTO_INDUSTRIAL": ["SERVICOS", np.nan],
        "DOMINIO_EMAIL": ["gmail.com"] * 2, "PORTE": ["PEQUENO"] * 2, "CEP_2_DIG": [1, 1],
    })
    i = pd.DataFrame({"ID_CLIENTE": [1, 2], "SAFRA_REF": ["2021-01", "2021-02"],
                      "RENDA_MES_ANTERIOR": [1000., 2000.], "NO_FUNCIONARIOS": [2, 3]})
    return c, i


class HistoricalAvailabilityTests(unittest.TestCase):
    def test_future_payments_do_not_change_past_features(self):
        raw = payments()
        changed = raw.copy()
        future = changed.DATA_PAGAMENTO.ge("2021-02-01")
        changed.loc[future, "DATA_PAGAMENTO"] += pd.Timedelta(days=50)
        a = s.add_history_features(raw, raw.iloc[2:])
        b = s.add_history_features(changed, raw.iloc[2:])
        assert_frame_equal(a[s.HISTORY_FEATURES], b[s.HISTORY_FEATURES])

    def test_binary_label_and_final_delay_have_different_availability(self):
        snapshot = s.history_asof(payments(), pd.Timestamp("2021-02-10")).set_index("ID_CLIENTE")
        self.assertEqual(snapshot.loc[1, "N_ROTULOS_CONHECIDOS"], 2)
        self.assertEqual(snapshot.loc[1, "N_PAGAMENTOS_CONHECIDOS"], 1)
        self.assertEqual(snapshot.loc[1, "ATRASO_MEDIO_HIST"], 2)
        self.assertEqual(snapshot.loc[1, "TAXA_INADIMPLENCIA_HIST"], .5)

    def test_event_on_cutoff_day_is_not_known(self):
        snapshot = s.history_asof(payments(), pd.Timestamp("2021-01-12"))
        self.assertEqual(snapshot.N_ROTULOS_CONHECIDOS.iloc[0], 0)

    def test_unknown_labels_do_not_enter_rate_denominator(self):
        snapshot = s.history_asof(payments(), pd.Timestamp("2021-02-01"))
        self.assertEqual(snapshot.N_COBRANCAS_ANTERIORES.iloc[0], 2)
        self.assertEqual(snapshot.N_ROTULOS_CONHECIDOS.iloc[0], 1)
        self.assertEqual(snapshot.TAXA_INADIMPLENCIA_HIST.iloc[0], 0)

    def test_training_and_scoring_share_value_mean(self):
        raw = payments()
        a = s.add_history_features_train(raw).iloc[2]
        b = s.add_history_features_test(raw.iloc[:2], raw.iloc[2:]).iloc[0]
        self.assertEqual(a.VALOR_MEDIO_HIST, 100.)
        self.assertEqual(b.VALOR_MEDIO_HIST, 100.)
        assert_frame_equal(s.add_history_features_train(raw).iloc[2:][s.HISTORY_FEATURES].reset_index(drop=True),
                           s.add_history_features_test(raw.iloc[:2], raw.iloc[2:])[s.HISTORY_FEATURES])

    def test_empty_history_counts_zero_means_unknown(self):
        result = s.add_history_features_train(payments())
        self.assertEqual(result.loc[0, "N_COBRANCAS_ANTERIORES"], 0)
        self.assertTrue(pd.isna(result.loc[0, "VALOR_MEDIO_HIST"]))
        self.assertTrue(pd.isna(result.loc[0, "INADIMPLENCIA_ULTIMA_SAFRA"]))

    def test_all_null_values_never_become_zero_mean(self):
        raw = payments().assign(VALOR_A_PAGAR=np.nan)
        self.assertTrue(s.add_history_features_train(raw).VALOR_MEDIO_HIST.isna().all())

    def test_input_order_is_preserved(self):
        raw = payments()
        rows = raw.iloc[[3, 0, 2, 1]].reset_index(drop=True)
        actual = s.add_history_features(raw, rows)
        assert_frame_equal(actual[rows.columns], rows)

    def test_snapshot_remains_frozen_across_future_months(self):
        raw = payments()
        rows = raw.iloc[[2]].copy()
        rows["SAFRA_REF"] = "2021-04"
        a = s.add_history_features(raw, rows, pd.Timestamp("2021-02-01"))
        self.assertEqual(a.N_COBRANCAS_ANTERIORES.iloc[0], 2)
        self.assertEqual(a.N_ROTULOS_CONHECIDOS.iloc[0], 1)


class PipelineContractTests(unittest.TestCase):
    def test_no_history_flag_is_learnable_in_training(self):
        raw = payments(); c, i = dimensions()
        frame = s.feature_table(c, i, raw, raw)
        self.assertEqual(frame.CLIENTE_SEM_HISTORICO.tolist(), [1, 1, 0, 1])

    def test_absent_registry_is_not_pj(self):
        raw = payments(); c, i = dimensions()
        raw.loc[0, "ID_CLIENTE"] = 99
        frame = s.add_profile_features(raw, c, i)
        self.assertEqual(frame.FLAG_PF.tolist(), ["DESCONHECIDO", "PJ", "PJ", "PF"])

    def test_future_payments_leave_training_membership_and_features_unchanged(self):
        raw = payments(); c, i = dimensions()
        changed = raw.copy()
        mask = changed.DATA_PAGAMENTO.ge("2021-02-01")
        changed.loc[mask, "DATA_PAGAMENTO"] += pd.Timedelta(days=30)
        a = s.training_table(c, i, raw, "2021-02-01")
        b = s.training_table(c, i, changed, "2021-02-01")
        assert_frame_equal(a[s.FEATURE_COLUMNS + ["INADIMPLENTE"]], b[s.FEATURE_COLUMNS + ["INADIMPLENTE"]])
        self.assertEqual(len(a), 1)

    def test_evaluation_includes_suspicious_dates(self):
        raw = payments(); c, i = dimensions()
        raw.loc[2, "DATA_PAGAMENTO"] = pd.Timestamp("2021-01-30")
        _, valid = s.prepare_validation_tables(c, i, raw, ["2021-02"])
        self.assertEqual(len(valid), 2)
        self.assertEqual(s.suspicious_dates(valid).sum(), 1)

    def test_null_keys_rejected_in_every_input(self):
        c, i = dimensions(); raw = payments(); test = raw.drop(columns="DATA_PAGAMENTO")
        for which in range(4):
            with self.subTest(which=which):
                args = [x.copy() for x in [c, i, raw, test]]
                args[which].loc[0, "ID_CLIENTE"] = np.nan
                with self.assertRaisesRegex(ValueError, "ID_CLIENTE nulo"):
                    s.validate_input_data(*args)

    def test_malformed_safra_rejected(self):
        c, i = dimensions(); raw = payments(); raw.loc[0, "SAFRA_REF"] = "2021-13"
        with self.assertRaisesRegex(ValueError, "SAFRA_REF invalida"):
            s.validate_input_data(c, i, raw, raw.drop(columns="DATA_PAGAMENTO"))

    def test_string_dates_rejected(self):
        c, i = dimensions(); raw = payments(); raw["DATA_PAGAMENTO"] = raw.DATA_PAGAMENTO.astype(str)
        with self.assertRaisesRegex(ValueError, "datetime"):
            s.validate_input_data(c, i, raw, raw.drop(columns="DATA_PAGAMENTO"))

    def test_random_early_stopping_is_disabled(self):
        self.assertFalse(s.make_hgb_pipeline(s.HGB_CONFIGS[0]).named_steps["model"].early_stopping)

    def test_calibration_freezes_complete_pipeline_and_uses_mature_labels(self):
        c, i = dimensions(); raw = payments()
        # Audit actual arguments without requiring a large synthetic HGB fit.
        class DummyModel:
            pass
        base = DummyModel()
        class FakeCalibrator:
            def __init__(self, estimator, method):
                self.estimator = estimator
            def fit(self, x, y):
                self.rows = len(x)
                return self
        raw.loc[3, "DATA_VENCIMENTO"] = pd.Timestamp("2021-02-02")
        raw.loc[3, "DATA_PAGAMENTO"] = pd.Timestamp("2021-02-03")
        with patch.object(s, "build_candidate_models", return_value={"hgb_cfg1_calibrado": base}), \
             patch.object(s, "fit_model"), patch.object(s, "FrozenEstimator", side_effect=lambda x: ("frozen", x)), \
             patch.object(s, "CalibratedClassifierCV", FakeCalibrator):
            model, details = s.fit_at_cutoff("hgb_cfg1_calibrado", c, i, raw, "2021-05-01")
        self.assertEqual(model.estimator, ("frozen", base))
        self.assertLess(details["train_latest_emission"], details["calibration_first_emission"])
        self.assertEqual(details["fit_cutoff"], "2021-02-01")
        self.assertEqual(model.rows, 2)


if __name__ == "__main__":
    unittest.main()
