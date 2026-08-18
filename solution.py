"""
Case Tecnico Datarisk - Cientista de Dados Junior

Executa a solucao de ponta a ponta:
1. carrega as bases CSV fornecidas;
2. remove do desenvolvimento cobrancas com data de vencimento inconsistente
   (rotulo nao confiavel -- ver filter_datas_inconsistentes);
3. constroi a variavel target na base de desenvolvimento;
4. cria features cadastrais, mensais, transacionais e historicas sem vazamento;
5. valida o modelo por safra temporal;
6. treina o modelo final e gera submissao_case.csv.

Uso:
    python solution.py             # grid padrao (~1-2 min)
    python solution.py --extended  # tambem reproduz peso balanceado e calibracao (~5 min)
    python solution.py --upload-supabase  # salva a submissao e publica as previsoes
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "submissao_case.csv"
RANDOM_STATE = 42
LIMIAR_ATRASO_DIAS = 5
VALIDATION_MONTHS = ("2021-03", "2021-04", "2021-05", "2021-06")

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
    return cadastral, info, pagamentos_dev, pagamentos_teste


def filter_datas_inconsistentes(pagamentos_dev: pd.DataFrame) -> pd.DataFrame:
    """Remove do desenvolvimento cobrancas com DATA_VENCIMENTO claramente inconsistente
    com DATA_EMISSAO_DOCUMENTO (prazo negativo ou maior que 400 dias -- mesmo criterio
    do PRAZO_ATIPICO usado como feature).

    Motivo: nessas linhas o ATRASO_DIAS (e portanto o rotulo INADIMPLENTE) e calculado
    sobre uma data quase certamente errada (ex.: erro de digitacao no ano). Isso nao e
    so um problema de feature: e um problema de ROTULO. Evidencia no desenvolvimento:
    essas linhas tem taxa de inadimplencia de ~59%, contra ~7% da base geral -- um
    padrao consistente com ruido de digitacao, nao com comportamento real de credito.
    Sem esse filtro, alem do rotulo da propria linha vir errado, os agregados
    historicos (TAXA_INADIMPLENCIA_HIST, ATRASO_MEDIO_HIST etc.) de safras futuras do
    mesmo cliente tambem ficariam contaminados por esses valores extremos.

    So se aplica ao desenvolvimento -- a base de teste nunca e filtrada, pois a
    submissao precisa cobrir 100% das linhas de `base_pagamentos_teste.csv`
    independentemente da qualidade dos dados; para o teste, PRAZO_ATIPICO continua
    disponivel como feature (nao como filtro).

    Chamada logo apos load_data() em main(), antes de qualquer construcao de
    target ou feature -- por isso fica posicionada aqui no arquivo, na mesma
    ordem em que roda de fato.
    """
    prazo = (pagamentos_dev["DATA_VENCIMENTO"] - pagamentos_dev["DATA_EMISSAO_DOCUMENTO"]).dt.days
    atipico = (prazo < 0) | (prazo > 400)
    removidas = int(atipico.sum())
    if removidas:
        print(
            f"\nRemovendo {removidas} cobranca(s) do desenvolvimento com prazo "
            f"emissao-vencimento inconsistente (<0 ou >400 dias) -- rotulo nao confiavel."
        )
    return pagamentos_dev.loc[~atipico].copy()


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
    return out


def ks_statistic(y_true: pd.Series, y_score: np.ndarray) -> float:
    scored = pd.DataFrame({"y": y_true.to_numpy(), "score": y_score}).sort_values("score", ascending=False)
    total_bad = scored["y"].sum()
    total_good = len(scored) - total_bad
    if total_bad == 0 or total_good == 0:
        return float("nan")
    cum_bad = scored["y"].cumsum() / total_bad
    cum_good = (1 - scored["y"]).cumsum() / total_good
    return float((cum_bad - cum_good).abs().max())


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
    out = df.merge(cadastral, on="ID_CLIENTE", how="left").merge(info, on=["ID_CLIENTE", "SAFRA_REF"], how="left")
    out["DATA_CADASTRO"] = pd.to_datetime(out["DATA_CADASTRO"], errors="coerce")
    out["DDD_NUM"] = pd.to_numeric(out["DDD"], errors="coerce")
    out["REGIAO"] = out["DDD_NUM"].round().astype("Int64").map(DDD_REGIAO)
    out["DIAS_COMO_CLIENTE"] = (out["DATA_EMISSAO_DOCUMENTO"] - out["DATA_CADASTRO"]).dt.days
    out["CADASTRO_ATIPICO"] = (out["DIAS_COMO_CLIENTE"] < 0).astype(int)
    out["DIAS_COMO_CLIENTE"] = out["DIAS_COMO_CLIENTE"].clip(lower=0)
    out["CADASTRO_AUSENTE"] = out["DATA_CADASTRO"].isna().astype(int)
    out["INFO_MENSAL_AUSENTE"] = out["RENDA_MES_ANTERIOR"].isna().astype(int)
    out["RENDA_LOG"] = np.log1p(out["RENDA_MES_ANTERIOR"].clip(lower=0))
    out["FUNCIONARIOS_LOG"] = np.log1p(out["NO_FUNCIONARIOS"].clip(lower=0))

    # Segundo o dicionario de dados, FLAG_PF e 'X' para pessoa fisica e NaN para
    # pessoa juridica -- ou seja, o NaN aqui NAO significa "desconhecido", significa
    # "e PJ". Se deixado como NaN, o imputer categorico trataria isso como uma
    # categoria generica "DESCONHECIDO", escondendo um sinal de risco real: no
    # desenvolvimento, clientes PF inadimplem a ~20% contra ~7% dos PJ.
    is_pf = out["FLAG_PF"] == "X"
    out["FLAG_PF"] = np.where(is_pf, "PF", "PJ")

    # SEGMENTO_INDUSTRIAL so existe para empresas (PJ); para PF, o NaN e estrutural
    # (nao se aplica), nao falta de dado. Separar os dois casos evita misturar
    # "cliente e pessoa fisica" com "empresa sem segmento cadastrado" na mesma
    # categoria "DESCONHECIDO".
    out.loc[is_pf, "SEGMENTO_INDUSTRIAL"] = "NAO_APLICAVEL_PF"

    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CLIENTE_SEM_HISTORICO"] = out["N_COBRANCAS_ANTERIORES"].isna().astype(int)
    out["VALOR_RENDA_RATIO"] = out["VALOR_A_PAGAR"] / out["RENDA_MES_ANTERIOR"].replace(0, np.nan)
    out["VALOR_RENDA_RATIO_CAP"] = out["VALOR_RENDA_RATIO"].clip(lower=0, upper=10)
    out["INADIMPLENCIA_ULTIMA_SAFRA"] = out["INADIMPLENCIA_ULTIMA_SAFRA"].fillna(0).astype(int)
    return out


def _historical_snapshot(dev_with_target: pd.DataFrame) -> pd.DataFrame:
    by_month = (
        dev_with_target.groupby(["ID_CLIENTE", "SAFRA_REF"], as_index=False)
        .agg(
            HIST_N_MES=("INADIMPLENTE", "size"),
            HIST_INAD_MES=("INADIMPLENTE", "sum"),
            HIST_ATRASO_SOMA_MES=("ATRASO_DIAS", "sum"),
            HIST_VALOR_SOMA_MES=("VALOR_A_PAGAR", "sum"),
            HIST_VALOR_MEDIO_MES=("VALOR_A_PAGAR", "mean"),
        )
        .sort_values(["ID_CLIENTE", "SAFRA_REF"])
    )
    grp = by_month.groupby("ID_CLIENTE", group_keys=False)
    by_month["N_COBRANCAS_ANTERIORES"] = grp["HIST_N_MES"].cumsum() - by_month["HIST_N_MES"]
    by_month["N_INADIMPLENCIAS_ANTERIORES"] = grp["HIST_INAD_MES"].cumsum() - by_month["HIST_INAD_MES"]
    by_month["ATRASO_SOMA_ANTERIOR"] = grp["HIST_ATRASO_SOMA_MES"].cumsum() - by_month["HIST_ATRASO_SOMA_MES"]
    by_month["VALOR_SOMA_ANTERIOR"] = grp["HIST_VALOR_SOMA_MES"].cumsum() - by_month["HIST_VALOR_SOMA_MES"]
    by_month["VALOR_MEDIO_MENSAL_ANTERIOR"] = grp["HIST_VALOR_MEDIO_MES"].shift(1)
    by_month["SAFRAS_OBSERVADAS_ANTERIORES"] = grp.cumcount()
    by_month["INADIMPLENCIA_ULTIMA_SAFRA"] = (grp["HIST_INAD_MES"].shift(1).fillna(0) > 0).astype(int)
    by_month["TAXA_INADIMPLENCIA_HIST"] = (
        by_month["N_INADIMPLENCIAS_ANTERIORES"] / by_month["N_COBRANCAS_ANTERIORES"].replace(0, np.nan)
    )
    by_month["ATRASO_MEDIO_HIST"] = (
        by_month["ATRASO_SOMA_ANTERIOR"] / by_month["N_COBRANCAS_ANTERIORES"].replace(0, np.nan)
    )
    by_month["VALOR_MEDIO_HIST"] = (
        by_month["VALOR_SOMA_ANTERIOR"] / by_month["N_COBRANCAS_ANTERIORES"].replace(0, np.nan)
    )
    cols = [
        "ID_CLIENTE",
        "SAFRA_REF",
        "N_COBRANCAS_ANTERIORES",
        "N_INADIMPLENCIAS_ANTERIORES",
        "SAFRAS_OBSERVADAS_ANTERIORES",
        "TAXA_INADIMPLENCIA_HIST",
        "ATRASO_MEDIO_HIST",
        "VALOR_MEDIO_HIST",
        "VALOR_MEDIO_MENSAL_ANTERIOR",
        "INADIMPLENCIA_ULTIMA_SAFRA",
    ]
    return by_month[cols]


def add_history_features_train(dev_with_target: pd.DataFrame) -> pd.DataFrame:
    """Anexa o historico do cliente a cada cobranca do DESENVOLVIMENTO.

    Anti-vazamento: usa janela expansiva -- para a cobranca de uma safra X, os
    agregados (TAXA_INADIMPLENCIA_HIST, ATRASO_MEDIO_HIST etc.) sao calculados
    apenas com safras estritamente anteriores a X do mesmo cliente (implementado
    via cumsum() - valor_do_mes_atual em _historical_snapshot). A propria safra
    nunca contribui para o seu proprio historico.
    """
    history = _historical_snapshot(dev_with_target)
    return dev_with_target.merge(history, on=["ID_CLIENTE", "SAFRA_REF"], how="left")


def add_history_features_test(dev_with_target: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Anexa o historico do cliente a cada cobranca do TESTE (ou da validacao).

    Diferente de add_history_features_train, aqui o historico e um snapshot
    UNICO por cliente -- agrega todo o `dev_with_target` recebido (sem janela
    expansiva por safra), porque a safra do teste e sempre posterior a todo o
    periodo de `dev_with_target` usado como argumento. Isso vale tanto para o
    teste real (recebe todo o desenvolvimento) quanto para a validacao temporal
    (recebe so o `train_raw`, que ja exclui os meses de validacao) -- em ambos
    os casos, nenhuma safra usada para calcular o historico e posterior a safra
    que esta sendo prevista.
    """
    summary = (
        dev_with_target.groupby("ID_CLIENTE", as_index=False)
        .agg(
            N_COBRANCAS_ANTERIORES=("INADIMPLENTE", "size"),
            N_INADIMPLENCIAS_ANTERIORES=("INADIMPLENTE", "sum"),
            SAFRAS_OBSERVADAS_ANTERIORES=("SAFRA_REF", "nunique"),
            TAXA_INADIMPLENCIA_HIST=("INADIMPLENTE", "mean"),
            ATRASO_MEDIO_HIST=("ATRASO_DIAS", "mean"),
            VALOR_MEDIO_HIST=("VALOR_A_PAGAR", "mean"),
        )
    )
    last_month_value = (
        dev_with_target.groupby(["ID_CLIENTE", "SAFRA_REF"], as_index=False)["VALOR_A_PAGAR"]
        .mean()
        .sort_values(["ID_CLIENTE", "SAFRA_REF"])
        .groupby("ID_CLIENTE", as_index=False)
        .tail(1)
        .rename(columns={"VALOR_A_PAGAR": "VALOR_MEDIO_MENSAL_ANTERIOR"})
        [["ID_CLIENTE", "VALOR_MEDIO_MENSAL_ANTERIOR"]]
    )
    last_month_inad = (
        dev_with_target.groupby(["ID_CLIENTE", "SAFRA_REF"], as_index=False)
        .agg(INADIMPLENCIA_ULTIMA_SAFRA=("INADIMPLENTE", "max"))
        .sort_values(["ID_CLIENTE", "SAFRA_REF"])
        .groupby("ID_CLIENTE", as_index=False)
        .tail(1)
        [["ID_CLIENTE", "INADIMPLENCIA_ULTIMA_SAFRA"]]
    )
    summary = summary.merge(last_month_value, on="ID_CLIENTE", how="left")
    summary = summary.merge(last_month_inad, on="ID_CLIENTE", how="left")
    return test.merge(summary, on="ID_CLIENTE", how="left")


def prepare_modeling_tables(
    cadastral: pd.DataFrame,
    info: pd.DataFrame,
    pagamentos_dev: pd.DataFrame,
    pagamentos_teste: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = add_basic_features(build_target(pagamentos_dev))
    test = add_basic_features(pagamentos_teste)
    dev = add_history_features_train(dev)
    test = add_history_features_test(dev, test)
    dev = add_profile_features(dev, cadastral, info)
    test = add_profile_features(test, cadastral, info)
    dev = add_derived_features(dev)
    test = add_derived_features(test)
    return dev, test


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
    "N_COBRANCAS_ANTERIORES",
    "N_INADIMPLENCIAS_ANTERIORES",
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


def make_hgb_pipeline(config: dict, calibrated: bool = False) -> Pipeline:
    hgb = HistGradientBoostingClassifier(
        learning_rate=config["learning_rate"],
        max_iter=config["max_iter"],
        max_leaf_nodes=config["max_leaf_nodes"],
        min_samples_leaf=config["min_samples_leaf"],
        l2_regularization=0.05,
        random_state=RANDOM_STATE,
    )
    if calibrated:
        model: object = CalibratedClassifierCV(hgb, method="isotonic", cv=3)
    else:
        model = hgb
    return Pipeline([("prep", make_preprocessor(scale_numeric=False)), (MODEL_STEP_NAME, model)])


def make_logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(scale_numeric=True)),
            (MODEL_STEP_NAME, LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
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
    auc = roc_auc_score(y_valid, proba)
    return {
        "modelo": name,
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_valid, proba),
        "brier": brier_score_loss(y_valid, proba),
        "prob_media": float(proba.mean()),
        "target_medio": float(y_valid.mean()),
    }


def temporal_split(df: pd.DataFrame, validation_months: Iterable[str] = VALIDATION_MONTHS):
    valid_mask = df["SAFRA_REF"].isin(list(validation_months))
    train_df = df.loc[~valid_mask].copy()
    valid_df = df.loc[valid_mask].copy()
    return train_df, valid_df


def prepare_validation_tables(
    cadastral: pd.DataFrame,
    info: pd.DataFrame,
    pagamentos_dev: pd.DataFrame,
    validation_months: Iterable[str] = VALIDATION_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev_basic = add_basic_features(build_target(pagamentos_dev))
    train_raw, valid_raw = temporal_split(dev_basic, validation_months=validation_months)
    train = add_history_features_train(train_raw)
    valid = add_history_features_test(train_raw, valid_raw)
    train = add_profile_features(train, cadastral, info)
    valid = add_profile_features(valid, cadastral, info)
    train = add_derived_features(train)
    valid = add_derived_features(valid)
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
    print("\nImportancia por permutacao (top 10)")
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
    )
    ranking = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": result.importances_mean}).sort_values(
        "importance", ascending=False
    )
    print(ranking.head(top_n).to_string(index=False, float_format=lambda x: f"{x:.5f}"))


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
    """Monta o dicionario de modelos candidatos.

    Por padrao (extended=False) roda apenas a baseline logistica e as 3
    configuracoes de HistGradientBoosting sem peso/calibracao -- o suficiente
    para reproduzir o modelo final em poucas dezenas de segundos.

    Com extended=True, tambem roda as variantes com sample_weight balanceado
    e as variantes calibradas (isotonic), reproduzindo a exploracao completa
    documentada no README. Essas variantes pioraram o Brier Score em todas as
    3 configuracoes testadas, entao nao fazem parte do pipeline padrao.
    """
    models: dict[str, Pipeline] = {"baseline_logistica": make_logistic_pipeline()}
    for idx, config in enumerate(HGB_CONFIGS, start=1):
        models[f"hgb_cfg{idx}"] = make_hgb_pipeline(config, calibrated=False)
        if extended:
            models[f"hgb_cfg{idx}_weighted"] = make_hgb_pipeline(config, calibrated=False)
            models[f"hgb_cfg{idx}_calibrado"] = make_hgb_pipeline(config, calibrated=True)
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case Tecnico Datarisk - treino e geracao da submissao")
    parser.add_argument(
        "--extended",
        action="store_true",
        help=(
            "Reproduz a exploracao completa (variantes com sample_weight balanceado e "
            "calibracao isotonic), alem do grid padrao. Mais lento (~5 min vs ~1 min) "
            "e nao muda o modelo final escolhido, apenas reproduz os experimentos "
            "documentados no README."
        ),
    )
    parser.add_argument(
        "--upload-supabase",
        action="store_true",
        help=(
            "Publica a execucao e as previsoes no Supabase depois de validar o CSV. "
            "Requer SUPABASE_URL e SUPABASE_SECRET_KEY no ambiente ou em .env."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cadastral, info, pagamentos_dev, pagamentos_teste = load_data()
    pagamentos_dev = filter_datas_inconsistentes(pagamentos_dev)
    dev, test = prepare_modeling_tables(cadastral, info, pagamentos_dev, pagamentos_teste)
    print_quality_summary(dev, test)
    print_eda_summary(dev)

    train_df, valid_df = prepare_validation_tables(cadastral, info, pagamentos_dev)
    x_train, y_train = train_df[FEATURE_COLUMNS], train_df["INADIMPLENTE"]
    x_valid, y_valid = valid_df[FEATURE_COLUMNS], valid_df["INADIMPLENTE"]

    print("\nValidacao temporal")
    print(f"Treino: {train_df['SAFRA_REF'].min()} a {train_df['SAFRA_REF'].max()} | {len(train_df):,} linhas")
    print(f"Validacao: {valid_df['SAFRA_REF'].min()} a {valid_df['SAFRA_REF'].max()} | {len(valid_df):,} linhas")
    if not args.extended:
        print("(grid padrao: baseline + 3 configs HGB. Use --extended para reproduzir tambem as")
        print(" variantes com peso balanceado e calibracao isotonic discutidas no README)")

    candidate_models = build_candidate_models(extended=args.extended)
    results: list[dict[str, float | str]] = []
    for name, model in candidate_models.items():
        fit_model(model, x_train, y_train, use_sample_weight=name.endswith("_weighted"))
        results.append(evaluate_model(name, model, x_valid, y_valid))

    results_df = pd.DataFrame(results).sort_values(["brier", "auc"], ascending=[True, False])
    print("\nMetricas na validacao")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_name = select_best_model(results_df)
    best_metrics = results_df.loc[results_df["modelo"] == best_name].iloc[0].to_dict()

    final_model = candidate_models[best_name]

    # A importancia por permutacao usa o modelo tal como foi avaliado na validacao
    # (ajustado so em x_train, ate 2021-02). Isso e proposital: x_valid (2021-03 a
    # 2021-06) faz parte do `dev` usado no refit final logo abaixo, entao calcular
    # a importancia DEPOIS do refit usaria dados que o modelo ja viu no treino
    # (nao seria mais held-out). Calculando aqui, x_valid continua genuinamente
    # nao visto pelo modelo avaliado.
    print_feature_importance(final_model, x_valid, y_valid)

    x_dev, y_dev = dev[FEATURE_COLUMNS], dev["INADIMPLENTE"]
    use_weights = best_name.endswith("_weighted")
    fit_model(final_model, x_dev, y_dev, use_sample_weight=use_weights)

    submission = pagamentos_teste[["ID_CLIENTE", "SAFRA_REF"]].copy()
    submission["PROBABILIDADE_INADIMPLENCIA"] = final_model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    submission["PROBABILIDADE_INADIMPLENCIA"] = submission["PROBABILIDADE_INADIMPLENCIA"].clip(0, 1)
    submission.to_csv(OUTPUT_FILE, sep=";", index=False)
    validate_submission(submission, expected_rows=len(pagamentos_teste))

    if args.upload_supabase:
        from supabase_io import upload_predictions

        run_id = upload_predictions(submission, model_name=best_name, metrics=best_metrics)
        print(f"Previsoes publicadas no Supabase. Run ID: {run_id}")

    print(f"\nModelo escolhido: {best_name}")
    print(f"Brier validacao: {best_metrics['brier']:.4f} | AUC validacao: {best_metrics['auc']:.4f}")
    print(f"Arquivo gerado: {OUTPUT_FILE.name}")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
