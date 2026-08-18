from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import pandas as pd

import supabase_io


class _Query:
    def __init__(self, calls: list[tuple[str, str, object]], table: str, action: str, payload: object = None):
        self.calls = calls
        self.table = table
        self.action = action
        self.payload = payload

    def eq(self, column: str, value: object):
        self.payload = (self.payload, column, value)
        return self

    def execute(self):
        self.calls.append((self.table, self.action, self.payload))
        return self


class _Table:
    def __init__(self, calls: list[tuple[str, str, object]], name: str):
        self.calls = calls
        self.name = name

    def insert(self, payload: object):
        return _Query(self.calls, self.name, "insert", payload)

    def delete(self):
        return _Query(self.calls, self.name, "delete")


class _Client:
    def __init__(self):
        self.calls: list[tuple[str, str, object]] = []

    def table(self, name: str):
        return _Table(self.calls, name)


class SupabaseIoTests(unittest.TestCase):
    def test_rejects_publishable_key(self):
        env = {
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_SECRET_KEY": "sb_publishable_example",
        }
        with patch.dict(os.environ, env, clear=True), self.assertRaisesRegex(RuntimeError, "chave publica"):
            supabase_io._client_from_env()

    def test_uploads_run_and_predictions(self):
        client = _Client()
        submission = pd.DataFrame(
            {
                "ID_CLIENTE": [101, 202],
                "SAFRA_REF": ["2021-07", "2021-08"],
                "PROBABILIDADE_INADIMPLENCIA": [0.1, 0.9],
            }
        )

        with patch.object(supabase_io, "_client_from_env", return_value=client):
            run_id = supabase_io.upload_predictions(
                submission,
                model_name="hgb_cfg3",
                metrics={"auc": 0.94, "brier": 0.03},
            )

        self.assertEqual(client.calls[0][0:2], ("prediction_runs", "insert"))
        self.assertEqual(client.calls[0][2]["id"], run_id)
        self.assertEqual(client.calls[1][0:2], ("credit_risk_predictions", "insert"))
        self.assertEqual(len(client.calls[1][2]), 2)
        self.assertEqual(client.calls[1][2][0]["client_id"], "101")


if __name__ == "__main__":
    unittest.main()
