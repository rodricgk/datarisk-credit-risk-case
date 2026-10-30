"""
Case Tecnico Datarisk - Cientista de Dados Junior

Executa a solucao de ponta a ponta:
1. carrega as bases CSV fornecidas;
2. separa elegibilidade de treino de populacao de avaliacao;
3. constroi a variavel target na base de desenvolvimento;
4. cria historicos com disponibilidade estrita no cutoff;
5. valida o modelo por safra temporal;
6. treina o modelo final e gera submissao_case.csv.

Uso:
    python solution.py             # grid padrao
    python solution.py --extended  # inclui peso balanceado e calibracao temporal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "submissao_case.csv"
RANDOM_STATE = 42
LIMIAR_ATRASO_DIAS = 5
VALIDATION_MONTHS = ("2021-03", "2021-04", "2021-05", "2021-06")
SELECTION_MONTHS = ("2020-11", "2020-12", "2021-01", "2021-02")

REQUIRED_COLUMNS = {
    "base_cadastral": {
        "ID_CLIENTE", "DATA_CADASTRO", "DDD", "FLAG_PF", "SEGMENTO_INDUSTRIAL",
        "DOMINIO_EMAIL", "PORTE", "CEP_2_DIG",
    },
    "base_info": {"ID_CLIENTE", "SAFRA_REF", "RENDA_MES_ANTERIOR", "NO_FUNCIONARIOS"},
    "pagamentos_dev": {
        "ID_CLIENTE", "SAFRA_REF", "DATA_EMISSAO_DOCUMENTO", "DATA_PAGAMENTO",
        "DATA_VENCIMENTO", "VALOR_A_PAGAR", "TAXA",
    },
    "pagamentos_teste": {
        "ID_CLIENTE", "SAFRA_REF", "DATA_EMISSAO_DOCUMENTO", "DATA_VENCIMENTO",
        "VALOR_A_PAGAR", "TAXA",
    },
}

HGB_CONFIGS = [
    {"max_leaf_nodes": 31, "min_samples_leaf": 20, "learning_rate": 0.05, "max_iter": 180},
    {"max_leaf_nodes": 63, "min_samples_leaf": 20, "learning_rate": 0.05, "max_iter": 180},
    {"max_leaf_nodes": 31, "min_samples_leaf": 50, "learning_rate": 0.03, "max_iter": 220},
]

DDD_REGIAO = {
    **{d: "Sudeste" for d in [11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 27, 28, 31, 32, 33, 34, 35, 37, 38]},
    **{d: "Sul" for d in [41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 53, 54, 55]},
    **{d: "Centro-Oeste" for d in [61, 62, 64, 65, 66, 67]},
    **{d: "Nordeste" for d in [71, 73, 74, 75, 77, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 98, 99]},
    **{d: "Norte" for d in [63, 68, 69, 91, 92, 93, 94, 95, 96, 97]},
}


# ============================================================
# Carregamento e limpeza de dados
# ============================================================


def load_data(base_dir: Path = BASE_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cadastral = pd.read_csv(base_dir / "base_cadastral.csv", sep=";")
    info = pd.read_csv(base_dir / "base_info.csv", sep=";")
    pagamentos_dev = pd.read_csv(
        base_dir / "base_pagamentos_desenvolvimento.csv",
        sep=";",
        parse_dates=["DATA_EMISSAO_DOCUMENTO", "DATA_VENCIMENTO", "DATA_PAGAMENTO"],
    )
    pagamentos_teste = pd.read_csv(
        base_dir / "base_pagamentos_teste.csv",
        sep=";",
        parse_dates=["DATA_EMISSAO_DOCUMENTO", "DATA_VENCIMENTO"],
    )
    validate_input_data(cadastral, info, pagamentos_dev, pagamentos_teste)
    return cadastral, info, pagamentos_dev, pagamentos_teste


def validate_input_data(
    cadastral: pd.DataFrame,
    info: pd.DataFrame,
    pagamentos_dev: pd.DataFrame,
    pagamentos_teste: pd.DataFrame,
) -> None:
    """Falha cedo quando esquema, chaves dimensionais ou datas essenciais mudam."""
    tables = {
        "base_cadastral": cadastral,
        "base_info": info,
        "pagamentos_dev": pagamentos_dev,
        "pagamentos_teste": pagamentos_teste,
    }
    for name, table in tables.items():
        missing = REQUIRED_COLUMNS[name] - set(table.columns)
        if missing:
            raise ValueError(f"{name}: colunas obrigatorias ausentes: {sorted(missing)}")
        if table.empty or table["ID_CLIENTE"].isna().any():
            raise ValueError(f"{name}: tabela vazia ou ID_CLIENTE nulo")
        if table["ID_CLIENTE"].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name}: ID_CLIENTE vazio")
        if "SAFRA_REF" in table:
            if not table["SAFRA_REF"].astype("string").str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").fillna(False).all():
                raise ValueError(f"{name}: SAFRA_REF invalida; esperado YYYY-MM")

    if cadastral.duplicated("ID_CLIENTE").any():
        raise ValueError("base_cadastral: ID_CLIENTE deve ser unico")
    if info.duplicated(["ID_CLIENTE", "SAFRA_REF"]).any():
        raise ValueError("base_info: (ID_CLIENTE, SAFRA_REF) deve ser unico")

    for name, table, date_columns in [
        ("pagamentos_dev", pagamentos_dev, ["DATA_EMISSAO_DOCUMENTO", "DATA_VENCIMENTO", "DATA_PAGAMENTO"]),
        ("pagamentos_teste", pagamentos_teste, ["DATA_EMISSAO_DOCUMENTO", "DATA_VENCIMENTO"]),
    ]:
        null_dates = table[date_columns].isna().sum()
        if null_dates.any():
            details = null_dates[null_dates > 0].to_dict()
            raise ValueError(f"{name}: datas essenciais ausentes: {details}")
        for column in date_columns:
            if not pd.api.types.is_datetime64_any_dtype(table[column]):
                raise ValueError(f"{name}: {column} precisa ser datetime valido")
        if not table["DATA_EMISSAO_DOCUMENTO"].dt.strftime("%Y-%m").eq(table["SAFRA_REF"]).all():
            raise ValueError(f"{name}: SAFRA_REF diverge do mes de emissao")
        for column in ["VALOR_A_PAGAR", "TAXA"]:
            values = pd.to_numeric(table[column], errors="coerce")
            if (table[column].notna() & values.isna()).any() or np.isinf(values).any():
                raise ValueError(f"{name}: {column} contem valor nao numerico ou infinito")


def filter_datas_inconsistentes(pagamentos_dev: pd.DataFrame) -> pd.DataFrame:
    """Compatibilidade: filtra SOMENTE treino; anomalias nao provam erro na origem.

    Nunca usar antes de materializar historicos ou selecionar avaliacao. A regra
    e uma hipotese conservadora de confiabilidade do rotulo, nao verdade factual.
    """
    return pagamentos_dev.loc[~suspicious_dates(pagamentos_dev)].copy()


def suspicious_dates(df: pd.DataFrame) -> pd.Series:
    """Diagnostico retrospectivo, nao regra para excluir linhas de avaliacao."""
    prazo = (df["DATA_VENCIMENTO"] - df["DATA_EMISSAO_DOCUMENTO"]).dt.days
    return (prazo < 0) | (prazo > 400) | (df["DATA_PAGAMENTO"] < df["DATA_EMISSAO_DOCUMENTO"])


# ============================================================
# Construcao da target e feature engineering
# ============================================================


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Regra de negocio do enunciado: INADIMPLENTE=1 quando o pagamento ocorre
    5 dias ou mais apos o vencimento (>=5, nao >5 -- fronteira validada manualmente
    com exemplos reais: atraso de 4 dias = adimplente, 5 dias = inadimplente)."""
    out = df.copy()
    out["ATRASO_DIAS"] = (out["DATA_PAGAMENTO"] - out["DATA_VENCIMENTO"]).dt.days
    out["INADIMPLENTE"] = (out["ATRASO_DIAS"] >= LIMIAR_ATRASO_DIAS).astype(int)
    threshold = out["DATA_VENCIMENTO"] + pd.Timedelta(days=LIMIAR_ATRASO_DIAS)
    out["LABEL_AVAILABLE_AT"] = pd.concat([out["DATA_PAGAMENTO"], threshold], axis=1).min(axis=1)
    return out


def ks_statistic(y_true: pd.Series, y_score: np.ndarray) -> float:
    if pd.Series(y_true).nunique() < 2:
        return float("nan")
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(true_positive_rate - false_positive_rate)))


def compute_sample_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    return y.map(lambda label: len(y) / (len(counts) * counts[label])).to_numpy()


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SAFRA_DATA"] = pd.to_datetime(out["SAFRA_REF"] + "-01")
    out["ANO_SAFRA"] = out["SAFRA_DATA"].dt.year
    out["MES_SAFRA"] = out["SAFRA_DATA"].dt.month
    out["MES_EMISSAO"] = out["DATA_EMISSAO_DOCUMENTO"].dt.month
    out["DIA_EMISSAO"] = out["DATA_EMISSAO_DOCUMENTO"].dt.day
    out["DIA_SEMANA_EMISSAO"] = out["DATA_EMISSAO_DOCUMENTO"].dt.dayofweek
    out["PRAZO_DIAS"] = (out["DATA_VENCIMENTO"] - out["DATA_EMISSAO_DOCUMENTO"]).dt.days
    out["PRAZO_ATIPICO"] = ((out["PRAZO_DIAS"] < 0) | (out["PRAZO_DIAS"] > 400)).astype(int)
    out["PRAZO_DIAS_CAP"] = out["PRAZO_DIAS"].clip(lower=0, upper=200)
    out["VALOR_A_PAGAR_LOG"] = np.log1p(out["VALOR_A_PAGAR"].clip(lower=0))
    return out


def add_profile_features(df: pd.DataFrame, cadastral: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_ROW_ORDER"] = np.arange(len(out))
    out = out.merge(
        cadastral, on="ID_CLIENTE", how="left", validate="many_to_one", indicator="_CADASTRO_MERGE", sort=False
    ).merge(
        info,
        on=["ID_CLIENTE", "SAFRA_REF"],
        how="left",
        validate="many_to_one",
        indicator="_INFO_MERGE",
        sort=False,
    )
    if len(out) != len(df):
        raise ValueError("Os joins cadastral/mensal alteraram a quantidade de cobrancas")
    out = out.sort_values("_ROW_ORDER").drop(columns="_ROW_ORDER").reset_index(drop=True)
    out["DATA_CADASTRO"] = pd.to_datetime(out["DATA_CADASTRO"], errors="coerce")
    out["DDD_NUM"] = pd.to_numeric(out["DDD"], errors="coerce")
    ddd_valido = out["DDD_NUM"].isin(DDD_REGIAO)
    out["DDD_AUSENTE"] = out["DDD"].isna().astype(int)
    out["DDD_INVALIDO"] = (out["DDD"].notna() & ~ddd_valido).astype(int)
    out["DDD"] = out["DDD_NUM"].where(ddd_valido).map(lambda value: f"{value:.0f}" if pd.notna(value) else "DESCONHECIDO")
    out["REGIAO"] = out["DDD_NUM"].where(ddd_valido).round().astype("Int64").map(DDD_REGIAO)
    out["DIAS_COMO_CLIENTE"] = (out["DATA_EMISSAO_DOCUMENTO"] - out["DATA_CADASTRO"]).dt.days
    out["CADASTRO_ATIPICO"] = (out["DIAS_COMO_CLIENTE"] < 0).astype(int)
    out["DIAS_COMO_CLIENTE"] = out["DIAS_COMO_CLIENTE"].clip(lower=0)
    out["CADASTRO_AUSENTE"] = (out["_CADASTRO_MERGE"] == "left_only").astype(int)
    out["INFO_MENSAL_AUSENTE"] = (out["_INFO_MERGE"] == "left_only").astype(int)
    out["RENDA_AUSENTE"] = out["RENDA_MES_ANTERIOR"].isna().astype(int)
    out["FUNCIONARIOS_AUSENTE"] = out["NO_FUNCIONARIOS"].isna().astype(int)
    out["RENDA_LOG"] = np.log1p(out["RENDA_MES_ANTERIOR"].clip(lower=0))
    out["FUNCIONARIOS_LOG"] = np.log1p(out["NO_FUNCIONARIOS"].clip(lower=0))

    # NaN significa PJ somente em uma linha cadastral existente.
    # Sem correspondencia no join, a natureza juridica e desconhecida.
    is_pf = out["FLAG_PF"] == "X"
    out["FLAG_PF"] = np.where(out["CADASTRO_AUSENTE"].eq(1), "DESCONHECIDO", np.where(is_pf, "PF", "PJ"))

    # SEGMENTO_INDUSTRIAL so existe para empresas (PJ); para PF, o NaN e estrutural
    # (nao se aplica), nao falta de dado. Separar os dois casos evita misturar
    # "cliente e pessoa fisica" com "empresa sem segmento cadastrado" na mesma
    # categoria "DESCONHECIDO".
    out.loc[is_pf, "SEGMENTO_INDUSTRIAL"] = "NAO_APLICAVEL_PF"
    return out.drop(columns=["_CADASTRO_MERGE", "_INFO_MERGE"])


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CLIENTE_SEM_HISTORICO"] = out["N_COBRANCAS_ANTERIORES"].fillna(0).eq(0).astype(int)
    out["VALOR_RENDA_RATIO"] = out["VALOR_A_PAGAR"] / out["RENDA_MES_ANTERIOR"].replace(0, np.nan)
    out["VALOR_RENDA_RATIO_CAP"] = out["VALOR_RENDA_RATIO"].clip(lower=0, upper=10)
    return out


HISTORY_COUNTS = [
    "N_COBRANCAS_ANTERIORES", "N_INADIMPLENCIAS_ANTERIORES",
    "SAFRAS_OBSERVADAS_ANTERIORES", "N_ROTULOS_CONHECIDOS", "N_PAGAMENTOS_CONHECIDOS",
]
HISTORY_FEATURES = HISTORY_COUNTS + [
    "TAXA_INADIMPLENCIA_HIST", "ATRASO_MEDIO_HIST", "VALOR_MEDIO_HIST",
    "VALOR_MEDIO_MENSAL_ANTERIOR", "INADIMPLENCIA_ULTIMA_SAFRA",
]


def history_asof(source: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Estado no inicio do dia: eventos no proprio cutoff ainda nao conhecidos.

    Contagens e valores usam todas as cobrancas emitidas antes do corte.
    Taxas usam somente rotulos maduros; atraso final exige pagamento observado.
    Datas suspeitas excluem somente os resultados, nao a existencia da cobranca.
    """
    past = source.loc[source["DATA_EMISSAO_DOCUMENTO"] < cutoff].copy()
    past = build_target(past)
    known = past["LABEL_AVAILABLE_AT"].lt(cutoff) & ~suspicious_dates(past)
    paid = past["DATA_PAGAMENTO"].lt(cutoff) & ~suspicious_dates(past)
    past["_known_target"] = past["INADIMPLENTE"].where(known)
    past["_known_delay"] = past["ATRASO_DIAS"].where(paid)
    summary = past.groupby("ID_CLIENTE").agg(
        N_COBRANCAS_ANTERIORES=("SAFRA_REF", "size"),
        N_INADIMPLENCIAS_ANTERIORES=("_known_target", "sum"),
        SAFRAS_OBSERVADAS_ANTERIORES=("SAFRA_REF", "nunique"),
        N_ROTULOS_CONHECIDOS=("_known_target", "count"),
        N_PAGAMENTOS_CONHECIDOS=("_known_delay", "count"),
        TAXA_INADIMPLENCIA_HIST=("_known_target", "mean"),
        ATRASO_MEDIO_HIST=("_known_delay", "mean"),
        VALOR_MEDIO_HIST=("VALOR_A_PAGAR", "mean"),
    )
    monthly = past.groupby(["ID_CLIENTE", "SAFRA_REF"], as_index=False).agg(
        VALOR_MEDIO_MENSAL_ANTERIOR=("VALOR_A_PAGAR", "mean"),
        INADIMPLENCIA_ULTIMA_SAFRA=("_known_target", "max"),
    ).sort_values(["ID_CLIENTE", "SAFRA_REF"]).groupby("ID_CLIENTE").tail(1)
    return summary.join(monthly.set_index("ID_CLIENTE").drop(columns="SAFRA_REF")).reset_index()


def add_history_features(
    source: pd.DataFrame, rows: pd.DataFrame, snapshot_cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Uma implementacao para treino, calibracao e scoring; preserva ordem."""
    out = rows.copy().reset_index(drop=True)
    cutoffs = pd.to_datetime(out["SAFRA_REF"] + "-01")
    if snapshot_cutoff is not None:
        cutoffs = cutoffs.clip(upper=pd.Timestamp(snapshot_cutoff))
    pieces = []
    for cutoff in sorted(cutoffs.unique()):
        group = out.loc[cutoffs.eq(cutoff)].copy()
        group["_order"] = group.index
        group = group.merge(history_asof(source, pd.Timestamp(cutoff)),
                            on="ID_CLIENTE", how="left", validate="many_to_one")
        pieces.append(group)
    if not pieces:
        raise ValueError("Nao ha cobrancas para construir historico")
    result = pd.concat(pieces).sort_values("_order").drop(columns="_order").reset_index(drop=True)
    result[HISTORY_COUNTS] = result[HISTORY_COUNTS].fillna(0)
    # Ultima safra sem rotulo conhecido permanece desconhecida, nunca vira zero.
    return result


def add_history_features_train(dev_with_target: pd.DataFrame) -> pd.DataFrame:
    return add_history_features(dev_with_target, dev_with_target)


def add_history_features_test(
    dev_with_target: pd.DataFrame, test: pd.DataFrame,
    snapshot_cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    cutoff = snapshot_cutoff if snapshot_cutoff is not None else pd.Timestamp(test["SAFRA_REF"].min() + "-01")
    return add_history_features(dev_with_target, test, cutoff)


def feature_table(cadastral, info, source, rows, snapshot_cutoff=None):
    rows = add_basic_features(rows)
    rows = add_history_features(source, rows, snapshot_cutoff)
    return add_derived_features(add_profile_features(rows, cadastral, info))


def training_table(cadastral, info, raw, cutoff, exclude_suspicious=True):
    cutoff = pd.Timestamp(cutoff)
    target = build_target(raw)
    eligible = target["DATA_EMISSAO_DOCUMENTO"].lt(cutoff) & target["LABEL_AVAILABLE_AT"].lt(cutoff)
    if exclude_suspicious:
        eligible &= ~suspicious_dates(target)
    rows = target.loc[eligible].sort_values(["SAFRA_REF", "DATA_EMISSAO_DOCUMENTO"])
    if rows.empty:
        raise ValueError("Treino sem rotulos maduros")
    # Historicos usam eventos brutos observaveis, nao so as linhas elegiveis ao fit.
    return feature_table(cadastral, info, raw, rows)


def prepare_modeling_tables(cadastral, info, pagamentos_dev, pagamentos_teste):
    cutoff = pd.Timestamp(pagamentos_teste["SAFRA_REF"].min() + "-01")
    if pagamentos_dev["SAFRA_REF"].max() >= pagamentos_teste["SAFRA_REF"].min():
        raise ValueError("Teste precisa ser posterior ao desenvolvimento")
    train = training_table(cadastral, info, pagamentos_dev, cutoff)
    test = feature_table(cadastral, info, pagamentos_dev, pagamentos_teste, cutoff)
    return train, test

NUMERIC_FEATURES = [
    "VALOR_A_PAGAR",
    "VALOR_A_PAGAR_LOG",
    "TAXA",
    "PRAZO_DIAS_CAP",
    "PRAZO_ATIPICO",
    "ANO_SAFRA",
    "MES_SAFRA",
    "MES_EMISSAO",
    "DIA_EMISSAO",
    "DIA_SEMANA_EMISSAO",
    "RENDA_MES_ANTERIOR",
    "RENDA_LOG",
    "NO_FUNCIONARIOS",
    "FUNCIONARIOS_LOG",
    "DIAS_COMO_CLIENTE",
    "CADASTRO_ATIPICO",
    "CADASTRO_AUSENTE",
    "INFO_MENSAL_AUSENTE",
    "RENDA_AUSENTE",
    "FUNCIONARIOS_AUSENTE",
    "DDD_AUSENTE",
    "DDD_INVALIDO",
    "N_COBRANCAS_ANTERIORES",
    "N_INADIMPLENCIAS_ANTERIORES",
    "N_ROTULOS_CONHECIDOS",
    "N_PAGAMENTOS_CONHECIDOS",
    "SAFRAS_OBSERVADAS_ANTERIORES",
    "TAXA_INADIMPLENCIA_HIST",
    "ATRASO_MEDIO_HIST",
    "VALOR_MEDIO_HIST",
    "VALOR_MEDIO_MENSAL_ANTERIOR",
    "CLIENTE_SEM_HISTORICO",
    "INADIMPLENCIA_ULTIMA_SAFRA",
    "VALOR_RENDA_RATIO_CAP",
]

CATEGORICAL_FEATURES = [
    "DDD",
    "REGIAO",
    "FLAG_PF",
    "SEGMENTO_INDUSTRIAL",
    "DOMINIO_EMAIL",
    "PORTE",
    "CEP_2_DIG",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# Definicao dos modelos
# ============================================================


def make_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="DESCONHECIDO")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


MODEL_STEP_NAME = "model"  # nome do step do Pipeline usado em fit_model() para rotear sample_weight


def make_hgb_pipeline(config: dict) -> Pipeline:
    hgb = HistGradientBoostingClassifier(
        learning_rate=config["learning_rate"],
        max_iter=config["max_iter"],
        max_leaf_nodes=config["max_leaf_nodes"],
        min_samples_leaf=config["min_samples_leaf"],
        l2_regularization=0.05,
        random_state=RANDOM_STATE,
        early_stopping=False,
    )
    return Pipeline([("prep", make_preprocessor(scale_numeric=False)), (MODEL_STEP_NAME, hgb)])


def make_logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(scale_numeric=True)),
            (MODEL_STEP_NAME, LogisticRegression(max_iter=1000, class_weight=None, random_state=RANDOM_STATE)),
        ]
    )


# ============================================================
# Treino e avaliacao
# ============================================================


def fit_model(model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, use_sample_weight: bool = False) -> Pipeline:
    if use_sample_weight:
        weights = compute_sample_weights(y_train)
        model.fit(x_train, y_train, **{f"{MODEL_STEP_NAME}__sample_weight": weights})
    else:
        model.fit(x_train, y_train)
    return model


def evaluate_model(name: str, model: Pipeline, x_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, float | str]:
    proba = model.predict_proba(x_valid)[:, 1]
    auc = roc_auc_score(y_valid, proba) if y_valid.nunique() > 1 else float("nan")
    brier = brier_score_loss(y_valid, proba)
    prevalence = float(y_valid.mean())
    reference_brier = prevalence * (1 - prevalence)
    return {
        "modelo": name,
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_valid, proba),
        "average_precision": average_precision_score(y_valid, proba),
        "brier": brier,
        "brier_skill": 1 - brier / reference_brier if reference_brier > 0 else float("nan"),
        "log_loss": log_loss(y_valid, proba, labels=[0, 1]),
        "prob_media": float(proba.mean()),
        "target_medio": prevalence,
    }


def temporal_split(df: pd.DataFrame, validation_months: Iterable[str] = VALIDATION_MONTHS):
    months = sorted(set(validation_months))
    if not months:
        raise ValueError("validation_months nao pode ser vazio")
    valid_mask = df["SAFRA_REF"].isin(months)
    train_mask = df["SAFRA_REF"] < months[0]
    unexpected = ~(valid_mask | train_mask)
    if unexpected.any():
        future_months = sorted(df.loc[unexpected, "SAFRA_REF"].unique())
        raise ValueError(f"Safras posteriores ou lacunas fora da validacao: {future_months}")
    train_df = df.loc[train_mask].copy()
    valid_df = df.loc[valid_mask].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("Treino e validacao temporal precisam conter observacoes")
    return train_df, valid_df


def prepare_validation_tables(
    cadastral: pd.DataFrame,
    info: pd.DataFrame,
    pagamentos_dev: pd.DataFrame,
    validation_months: Iterable[str] = VALIDATION_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(min(validation_months) + "-01")
    train = training_table(cadastral, info, pagamentos_dev, cutoff)
    rows = build_target(pagamentos_dev.loc[pagamentos_dev["SAFRA_REF"].isin(validation_months)])
    valid = feature_table(cadastral, info, pagamentos_dev, rows, cutoff)
    return train, valid


# ============================================================
# Relatorios (EDA, qualidade de dados, importancia de variaveis)
# ============================================================


def print_eda_summary(dev: pd.DataFrame) -> None:
    print("\nEDA resumida")
    print("Inadimplencia por safra (ultimas 6):")
    by_safra = dev.groupby("SAFRA_REF")["INADIMPLENTE"].mean().tail(6)
    for safra, taxa in by_safra.items():
        print(f"  {safra}: {taxa:.4f}")

    print("\nInadimplencia por porte:")
    for porte, taxa in dev.groupby("PORTE")["INADIMPLENTE"].mean().sort_values(ascending=False).head(5).items():
        print(f"  {porte}: {taxa:.4f}")

    print("\nInadimplencia por regiao:")
    for regiao, taxa in dev.groupby("REGIAO")["INADIMPLENTE"].mean().sort_values(ascending=False).items():
        print(f"  {regiao}: {taxa:.4f}")

    print("\nCobertura de dados:")
    print(f"  Cadastro ausente: {dev['CADASTRO_AUSENTE'].mean():.4f}")
    print(f"  Info mensal ausente: {dev['INFO_MENSAL_AUSENTE'].mean():.4f}")
    print(f"  Prazo atipico: {dev['PRAZO_ATIPICO'].mean():.4f}")
    print(f"  Cliente sem historico: {dev['CLIENTE_SEM_HISTORICO'].mean():.4f}")


def print_quality_summary(dev: pd.DataFrame, test: pd.DataFrame) -> None:
    print("\nResumo dos dados")
    print(f"Desenvolvimento: {len(dev):,} linhas | {dev['ID_CLIENTE'].nunique():,} clientes")
    print(f"Teste: {len(test):,} linhas | {test['ID_CLIENTE'].nunique():,} clientes")
    print(f"Taxa de inadimplencia no desenvolvimento: {dev['INADIMPLENTE'].mean():.4f}")
    print(f"Safras desenvolvimento: {dev['SAFRA_REF'].min()} a {dev['SAFRA_REF'].max()}")
    print(f"Safras teste: {test['SAFRA_REF'].min()} a {test['SAFRA_REF'].max()}")
    print(f"Linhas de teste com cliente sem historico no desenvolvimento: {test['CLIENTE_SEM_HISTORICO'].sum():,}")
    print(f"Linhas de dev com prazo atipico: {int(dev['PRAZO_ATIPICO'].sum()):,}")
    print(f"Linhas de teste com prazo atipico: {int(test['PRAZO_ATIPICO'].sum()):,}")


def print_feature_importance(model: Pipeline, x_valid: pd.DataFrame, y_valid: pd.Series, top_n: int = 10) -> None:
    print("\nImportancia por permutacao em Brier Score (top 10)")
    sample_n = min(3000, len(x_valid))
    sample_idx = x_valid.sample(n=sample_n, random_state=RANDOM_STATE).index
    x_sample = x_valid.loc[sample_idx]
    y_sample = y_valid.loc[sample_idx]
    result = permutation_importance(
        model,
        x_sample,
        y_sample,
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
        scoring="neg_brier_score",
    )
    ranking = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": result.importances_mean}).sort_values(
        "importance", ascending=False
    )
    print(ranking.head(top_n).to_string(index=False, float_format=lambda x: f"{x:.5f}"))


def print_monthly_metrics(model: Pipeline, valid_df: pd.DataFrame) -> None:
    """Explicita degradacao temporal que a metrica agregada pode esconder."""
    rows = []
    for month, group in valid_df.groupby("SAFRA_REF", sort=True):
        metrics = evaluate_model(
            str(month), model, group[FEATURE_COLUMNS], group["INADIMPLENTE"]
        )
        metrics["safra"] = month
        metrics["n"] = len(group)
        rows.append(metrics)
    columns = ["safra", "n", "target_medio", "prob_media", "auc", "ks", "average_precision", "brier"]
    print("\nMetricas por safra de validacao")
    print(pd.DataFrame(rows)[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ============================================================
# Execucao principal
# ============================================================


def select_best_model(results_df: pd.DataFrame) -> str:
    ordered = results_df.sort_values(["brier", "auc"], ascending=[True, False])
    return str(ordered.iloc[0]["modelo"])


def validate_submission(submission: pd.DataFrame, expected_rows: int) -> None:
    """Confere em tempo de execucao os requisitos do enunciado para submissao_case.csv."""
    expected_cols = ["ID_CLIENTE", "SAFRA_REF", "PROBABILIDADE_INADIMPLENCIA"]
    assert list(submission.columns) == expected_cols, f"Colunas inesperadas: {list(submission.columns)}"
    assert len(submission) == expected_rows, f"Esperava {expected_rows:,} linhas, gerou {len(submission):,}"
    assert submission.isna().sum().sum() == 0, "Existem valores nulos na submissao"
    p = submission["PROBABILIDADE_INADIMPLENCIA"]
    assert p.between(0, 1).all(), f"Probabilidades fora de [0, 1]: min={p.min()}, max={p.max()}"
    print(f"\nValidacao da submissao: OK ({len(submission):,} linhas, colunas corretas, sem nulos, probabilidades em [0, 1])")


def build_candidate_models(extended: bool) -> dict[str, Pipeline]:
    models = {"baseline_logistica": make_logistic_pipeline()}
    for idx, config in enumerate(HGB_CONFIGS, start=1):
        models[f"hgb_cfg{idx}"] = make_hgb_pipeline(config)
        if extended:
            models[f"hgb_cfg{idx}_weighted"] = make_hgb_pipeline(config)
            models[f"hgb_cfg{idx}_calibrado"] = make_hgb_pipeline(config)
    return models


def fit_at_cutoff(name, cadastral, info, raw, cutoff, exclude_suspicious=True):
    """Mesmo protocolo no experimento e no refit: maturacao e holdout por safra."""
    cutoff = pd.Timestamp(cutoff)
    calibrated = name.endswith("_calibrado")
    fit_cutoff = cutoff - pd.DateOffset(months=3) if calibrated else cutoff
    train = training_table(cadastral, info, raw, fit_cutoff, exclude_suspicious)
    model = build_candidate_models(extended=True)[name]
    fit_model(model, train[FEATURE_COLUMNS], train["INADIMPLENTE"], name.endswith("_weighted"))
    details = {"fit_cutoff": str(fit_cutoff.date()), "label_cutoff": str(cutoff.date()),
               "train_n": len(train), "train_latest_emission": str(train.DATA_EMISSAO_DOCUMENTO.max().date()),
               "calibration_n": 0}
    if calibrated:
        targets = build_target(raw)
        mask = targets.DATA_EMISSAO_DOCUMENTO.ge(fit_cutoff) & targets.DATA_EMISSAO_DOCUMENTO.lt(cutoff)
        mask &= targets.LABEL_AVAILABLE_AT.lt(cutoff)
        if exclude_suspicious:
            mask &= ~suspicious_dates(targets)
        calibration = feature_table(cadastral, info, raw, targets.loc[mask], fit_cutoff)
        if calibration.INADIMPLENTE.nunique() != 2:
            raise ValueError("Calibracao temporal precisa de ambas as classes")
        # Congela TODO o pipeline, inclusive imputacao e one-hot. Nao ha folds
        # por posicao, nem ajuste do pre-processador nas observacoes de calibracao.
        model = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
        model.fit(calibration[FEATURE_COLUMNS], calibration.INADIMPLENTE)
        details["calibration_n"] = len(calibration)
        details["calibration_first_emission"] = str(calibration.DATA_EMISSAO_DOCUMENTO.min().date())
    return model, details


def metric_row(model, frame, name):
    row = evaluate_model(name, model, frame[FEATURE_COLUMNS], frame.INADIMPLENTE)
    return {"n": len(frame), "events": int(frame.INADIMPLENTE.sum()), **row}


def client_bootstrap(frame, scores, repeats=300, seed=RANDOM_STATE):
    """IC condicional ao modelo fixo; nao inclui selecao ou mudancas de regime."""
    rng = np.random.default_rng(seed)
    groups = list(frame.groupby("ID_CLIENTE", sort=True).indices.values())
    values = []
    for _ in range(repeats):
        indices = np.concatenate([groups[k] for k in rng.integers(len(groups), size=len(groups))])
        y, p = frame.INADIMPLENTE.iloc[indices], scores[indices]
        if y.nunique() == 2:
            values.append([roc_auc_score(y, p), brier_score_loss(y, p)])
    if not values:
        raise ValueError("Bootstrap sem amostras com duas classes")
    quantiles = np.quantile(values, [.025, .975], axis=0)
    return {"repeats_requested": repeats, "repeats_valid": len(values), "seed": seed,
            "auc_95": quantiles[:, 0].tolist(), "brier_95": quantiles[:, 1].tolist()}


def run_experiment(cadastral, info, raw, test_raw, extended=False):
    """Selecao Nov/20-Fev/21; auditoria Mar-Jun/21; refit Jul/21. Nenhum CSV filtrado."""
    audit_cutoff = pd.Timestamp(min(VALIDATION_MONTHS) + "-01")
    selection_cutoff = pd.Timestamp(min(SELECTION_MONTHS) + "-01")
    targets = build_target(raw)
    selection_mask = targets.SAFRA_REF.isin(SELECTION_MONTHS)
    selection_eligible = selection_mask & targets.LABEL_AVAILABLE_AT.lt(audit_cutoff)
    selection = feature_table(cadastral, info, raw, targets.loc[selection_eligible], selection_cutoff)
    audit = feature_table(cadastral, info, raw, targets.loc[targets.SAFRA_REF.isin(VALIDATION_MONTHS)], audit_cutoff)
    results, fits = [], {}
    for name in build_candidate_models(extended):
        print(f"Selecao: {name}", flush=True)
        model, details = fit_at_cutoff(name, cadastral, info, raw, selection_cutoff)
        results.append(metric_row(model, selection, name))
        fits[name] = details
    selected = select_best_model(pd.DataFrame(results))
    print(f"Selecionado sem consultar Mar-Jun: {selected}", flush=True)
    model, audit_fit = fit_at_cutoff(selected, cadastral, info, raw, audit_cutoff)
    scores = model.predict_proba(audit[FEATURE_COLUMNS])[:, 1]
    monthly = [metric_row(model, group, month) for month, group in audit.groupby("SAFRA_REF")]
    cohorts = []
    for label, mask in [("sem_historico", audit.CLIENTE_SEM_HISTORICO.eq(1)),
                        ("com_historico", audit.CLIENTE_SEM_HISTORICO.eq(0)),
                        ("datas_suspeitas", suspicious_dates(audit)),
                        ("demais_datas", ~suspicious_dates(audit))]:
        if mask.any():
            group = audit.loc[mask].reset_index(drop=True)
            row = metric_row(model, group, label)
            if group.INADIMPLENTE.nunique() == 2:
                row["bootstrap"] = client_bootstrap(group, scores[mask])
            cohorts.append(row)
    bins = pd.DataFrame({"score": scores, "target": audit.INADIMPLENTE.to_numpy()})
    bins["faixa"] = pd.qcut(bins.score, 10, duplicates="drop")
    reliability = bins.groupby("faixa", observed=True).agg(
        n=("target", "size"), predicted=("score", "mean"), observed=("target", "mean")
    ).reset_index()
    reliability["faixa"] = reliability["faixa"].astype(str)
    # Sensibilidade do filtro SOMENTE no fit, sem trocar a populacao de avaliacao.
    sensitivity_model, _ = fit_at_cutoff(selected, cadastral, info, raw, audit_cutoff, exclude_suspicious=False)
    sensitivity = metric_row(sensitivity_model, audit, "treino_inclui_datas_suspeitas")
    final_cutoff = pd.Timestamp(test_raw.SAFRA_REF.min() + "-01")
    if raw.SAFRA_REF.max() >= test_raw.SAFRA_REF.min():
        raise ValueError("Teste precisa ser posterior ao desenvolvimento")
    final_model, final_fit = fit_at_cutoff(selected, cadastral, info, raw, final_cutoff)
    test = feature_table(cadastral, info, raw, test_raw, final_cutoff)
    submission = test_raw[["ID_CLIENTE", "SAFRA_REF"]].reset_index(drop=True).copy()
    submission["PROBABILIDADE_INADIMPLENCIA"] = final_model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    validate_submission(submission, len(test_raw))
    report = {
        "protocol": "monthly-asof-v2", "extended": extended, "selected": selected,
        "population": {"development": len(raw), "suspicious_development": int(suspicious_dates(raw).sum()),
                       "selection_total": int(selection_mask.sum()), "selection_mature": len(selection),
                       "selection_unmature_excluded": int(selection_mask.sum()) - len(selection),
                       "audit": len(audit), "audit_excluded": 0, "test": len(test_raw)},
        "selection": results, "selection_fits": fits, "audit_fit": audit_fit,
        "audit": metric_row(model, audit, selected), "monthly": monthly, "cohorts": cohorts,
        "reliability": reliability.to_dict("records"),
        "bootstrap": client_bootstrap(audit, scores),
        "sensitivity_same_audit_population": sensitivity, "final_fit": final_fit,
    }
    return report, submission


def markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(
            f"{row[col]:.6f}" if isinstance(row.get(col), float) else str(row.get(col, ""))
            for col in columns) + " |")
    return "\n".join(lines)


def save_report(report, data_dir, report_dir):
    """Fonte unica para notebook, tabelas publicadas e parametros do experimento."""
    import importlib.metadata as metadata
    report["versions"] = {package: metadata.version(package) for package in ["numpy", "pandas", "scikit-learn"]}
    report["data_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(data_dir).glob("base*.csv"))
    }
    report["solution_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    # pandas serializa NaN como null, mantendo JSON estrito.
    (report_dir / "metrics.json").write_text(
        pd.Series(report, dtype=object).to_json(force_ascii=False, indent=2, double_precision=12), encoding="utf-8")
    columns = ["modelo", "n", "auc", "ks", "average_precision", "brier", "prob_media", "target_medio"]
    content = [
        "# Resultados reproduzidos — protocolo temporal corrigido",
        "Gerado por solution.py. Fonte dos valores e metadados: [metrics.json](metrics.json).",
        "## Seleção: novembro/2020 a fevereiro/2021",
        "Somente rótulos disponíveis antes de 01/03/2021 participam da seleção.",
        markdown_table(report["selection"], columns),
        "## Auditoria: março a junho/2021 — todas as cobranças",
        "Configuração escolhida na janela anterior; histórico congelado em 01/03/2021.",
        "**Ressalva:** este período foi examinado nas versões antigas. Não é um teste prospectivo cego.",
        markdown_table([report["audit"]], columns),
        "## Por safra", markdown_table(report["monthly"], columns),
        "## Segmentos", markdown_table(report["cohorts"], columns),
        "Os ICs por cliente dos segmentos com duas classes estão no JSON. Grupos pequenos exigem cautela.",
        "## Confiabilidade por decil de score",
        markdown_table(report["reliability"], ["faixa", "n", "predicted", "observed"]),
        "Brier não isola calibração; as diferenças observado-previsto são diagnósticos descritivos.",
        "## Sensibilidade: incluir datas suspeitas somente no treino",
        "A população de auditoria permanece idêntica; não se usa o resultado para escolher outra configuração.",
        markdown_table([report["sensitivity_same_audit_population"]], columns),
        "## Incerteza",
        f"Bootstrap por cliente: {report['bootstrap']}. Condicional ao modelo; não mede seleção nem regimes futuros.",
        "## População", str(report["population"]),
        "## Treino final", str(report["final_fit"]),
        "Sem alegação de ganho financeiro, calibração perfeita ou ausência de todos os vieses."
    ]
    (report_dir / "metrics.md").write_text("\n\n".join(content) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case Datarisk: experimento temporal e submissao")
    parser.add_argument("--extended", action="store_true", help="Inclui pesos e calibracao temporal; vencedor pode mudar.")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATARISK_DATA_DIR", BASE_DIR)))
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--report-dir", type=Path, default=BASE_DIR / "reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cadastral, info, raw, test = load_data(args.data_dir)
    report, submission = run_experiment(cadastral, info, raw, test, args.extended)
    save_report(report, args.data_dir, args.report_dir)
    submission.to_csv(args.output, sep=";", index=False)
    print(json.dumps(report["audit"], indent=2))
    print(f"Submissao: {args.output}; relatorio: {args.report_dir}")


if __name__ == "__main__":
    main()
