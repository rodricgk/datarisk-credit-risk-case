"""Persistencia opcional das previsoes do case no Supabase.

Este modulo e carregado apenas quando ``--upload-supabase`` e usado. A chave
secreta fica no ambiente local e nunca deve ser adicionada ao repositorio.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client


BASE_DIR = Path(__file__).resolve().parent
BATCH_SIZE = 500


def _client_from_env() -> Client:
    load_dotenv(BASE_DIR / ".env")
    url = os.getenv("SUPABASE_URL", "").strip()
    secret_key = os.getenv("SUPABASE_SECRET_KEY", "").strip()

    missing = [name for name, value in (("SUPABASE_URL", url), ("SUPABASE_SECRET_KEY", secret_key)) if not value]
    if missing:
        raise RuntimeError(
            "Configuracao Supabase ausente: "
            + ", ".join(missing)
            + ". Copie .env.example para .env e preencha somente no ambiente local."
        )
    if secret_key.startswith(("sb_publishable_", "eyJ")):
        raise RuntimeError(
            "SUPABASE_SECRET_KEY deve receber uma chave secreta sb_secret_* "
            "(ou a service_role legada), nao uma chave publica/anon."
        )

    return create_client(url, secret_key)


def upload_predictions(
    submission: pd.DataFrame,
    *,
    model_name: str,
    metrics: Mapping[str, object],
) -> str:
    """Cria uma execucao e envia suas previsoes em lotes.

    Em caso de falha parcial, tenta remover a execucao e os registros ja
    enviados antes de propagar o erro original.
    """
    client = _client_from_env()
    run_id = str(uuid4())
    run = {
        "id": run_id,
        "model_name": model_name,
        "validation_auc": float(metrics["auc"]),
        "validation_brier": float(metrics["brier"]),
        "prediction_count": int(len(submission)),
    }
    rows = [
        {
            "run_id": run_id,
            "prediction_order": int(order),
            "client_id": str(row.ID_CLIENTE),
            "vintage": str(row.SAFRA_REF),
            "default_probability": float(row.PROBABILIDADE_INADIMPLENCIA),
        }
        for order, row in enumerate(submission.itertuples(index=False))
    ]

    try:
        client.table("prediction_runs").insert(run).execute()
        for start in range(0, len(rows), BATCH_SIZE):
            client.table("credit_risk_predictions").insert(rows[start : start + BATCH_SIZE]).execute()
    except Exception:
        # A chave secreta tem permissao para limpar uma carga incompleta.
        try:
            client.table("credit_risk_predictions").delete().eq("run_id", run_id).execute()
            client.table("prediction_runs").delete().eq("id", run_id).execute()
        except Exception:
            pass
        raise

    return run_id
