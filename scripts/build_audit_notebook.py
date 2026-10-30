"""Gera e executa o notebook de auditoria. Requer DATARISK_DATA_DIR."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
notebook = nbf.v4.new_notebook(cells=[
    md("""# Auditoria de qualidade e validação — Case Datarisk

## tl;dr

Esta execução substitui as métricas do protocolo anterior. As tabelas abaixo são
calculadas do código corrigido, não transcritas do README. O modelo é escolhido
em novembro/2020–fevereiro/2021 e auditado em março–junho/2021, sem excluir
cobranças da auditoria. O período de auditoria já foi examinado anteriormente:
não é um teste prospectivo cego.

## Context & Methods

Unidade: cobrança; incerteza reamostrada por cliente. Seleção pelo Brier.
Calibração opcional: três meses completos anteriores ao corte, com pipeline
congelado e somente rótulos maduros. Early stopping aleatório desativado.

### Key Assumptions

- Previsão na emissão; histórico até o início da safra, limitado ao início do bloco.
- Datas sem hora: só eventos estritamente anteriores ao corte são conhecidos.
- Alvo conhecido no pagamento ou em vencimento + 5 dias; atraso final só no pagamento.
- Cadastro estático e informação mensal disponíveis na emissão são hipóteses:
  a fonte não fornece versões históricas nem timestamps de ingestão.
- Datas suspeitas são hipótese de qualidade, não erros comprovados. Excluídas
  do ajuste dos rótulos, mas não da população de auditoria ou teste.
- ICs condicionais ao modelo; não abrangem seleção nem mudanças de regime.

Fonte: [case oficial](https://github.com/datarisk-io/datarisk-case-ds-junior).
Defina DATARISK_DATA_DIR para a pasta dos quatro CSVs; não são republicados.
"""),
    code("""from pathlib import Path
import os, sys, hashlib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / 'solution.py').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import solution as s
DATA_DIR = Path(os.environ.get('DATARISK_DATA_DIR', ROOT))
EXTENDED = True
cadastral, info, raw, test_raw = s.load_data(DATA_DIR)
"""),
    md("## Data\n\nGrão, nulos e fontes antes de filtrar qualquer população."),
    code("""profile = pd.DataFrame([
    {'base': name, 'linhas': len(frame), 'clientes': frame.ID_CLIENTE.nunique(),
     'celulas_nulas': int(frame.isna().sum().sum())}
    for name, frame in [('cadastro', cadastral), ('mensal', info), ('desenvolvimento', raw), ('teste', test_raw)]
])
display(profile)
display(pd.Series({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(DATA_DIR.glob('base*.csv'))}, name='sha256'))
"""),
    md("### Regressões nas bases completas\n\nEventos futuros não podem mudar features passadas ou rótulos elegíveis."),
    code("""train, audit = s.prepare_validation_tables(cadastral, info, raw)
cutoff = pd.Timestamp('2021-03-01')
changed = raw.copy()
future = changed.DATA_PAGAMENTO.ge(cutoff)
changed.loc[future, 'DATA_PAGAMENTO'] += pd.Timedelta(days=31)
train_changed, audit_changed = s.prepare_validation_tables(cadastral, info, changed)
assert_frame_equal(audit[s.FEATURE_COLUMNS], audit_changed[s.FEATURE_COLUMNS])
assert_frame_equal(train[s.FEATURE_COLUMNS + ['INADIMPLENTE']],
                   train_changed[s.FEATURE_COLUMNS + ['INADIMPLENTE']])
assert train.LABEL_AVAILABLE_AT.lt(cutoff).all()
assert len(audit) == raw.SAFRA_REF.isin(s.VALIDATION_MONTHS).sum()
assert train.CLIENTE_SEM_HISTORICO.sum() == train.N_COBRANCAS_ANTERIORES.eq(0).sum() > 0
test_features = s.feature_table(cadastral, info, raw, test_raw, pd.Timestamp('2021-07-01'))
assert test_features.loc[test_features.CADASTRO_AUSENTE.eq(1), 'FLAG_PF'].eq('DESCONHECIDO').all()
display(pd.Series({'treino_maduro': len(train), 'auditoria_sem_exclusoes': len(audit),
                   'sem_historico_no_treino': int(train.CLIENTE_SEM_HISTORICO.sum()),
                   'cadastro_ausente_teste': int(test_features.CADASTRO_AUSENTE.sum()),
                   'invariancia_futura': 'aprovada'}))
"""),
    md("## Results\n\nGrade completa, avaliação separada, sensibilidade e submissão local."),
    code("""report, submission = s.run_experiment(cadastral, info, raw, test_raw, extended=EXTENDED)
s.save_report(report, DATA_DIR, ROOT / 'reports')
submission.to_csv(ROOT / 'submissao_case.csv', sep=';', index=False)
assert np.isfinite(submission.PROBABILIDADE_INADIMPLENCIA).all()
assert_frame_equal(submission[['ID_CLIENTE', 'SAFRA_REF']],
                   test_raw[['ID_CLIENTE', 'SAFRA_REF']].reset_index(drop=True))
display(pd.DataFrame(report['selection']).sort_values('brier'))
display(pd.DataFrame([report['audit']]))
"""),
    md("### Safras, segmentos e calibração\n\nBrier não isola calibração. Observe também taxas e probabilidades médias; grupos pequenos são incertos."),
    code("""display(pd.DataFrame(report['monthly']))
display(pd.DataFrame(report['cohorts']).drop(columns='bootstrap', errors='ignore'))
display(pd.DataFrame(report['reliability']))
display(pd.Series(report['bootstrap'], name='bootstrap_por_cliente'))
display(pd.DataFrame([report['sensitivity_same_audit_population']]))
"""),
    md("## Takeaways\n\nAs conclusões abaixo são geradas dos resultados executados."),
    code("""from IPython.display import Markdown
a = report['audit']
display(Markdown(f\"Modelo selecionado: **{report['selected']}**. Auditoria com **{a['n']} cobranças**: \"
                 f\"AUC **{a['auc']:.4f}**, Brier **{a['brier']:.5f}**, AP **{a['average_precision']:.4f}**.\"))
display(Markdown('Os testes de regressão passaram. Os resultados continuam condicionados às hipóteses '
                 'de disponibilidade cadastral/mensal e à qualidade dos rótulos oficiais. '
                 'Não comparar estes números como ganho ou perda isolada de algoritmo: '
                 'mudaram informação disponível, população, seleção e treino. '
                 'Um período novo é necessário para confirmação prospectiva.'))
"""),
], metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
path = ROOT / "notebooks" / "data_quality_audit.ipynb"
nbf.write(notebook, path)
NotebookClient(notebook, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.validate(notebook)
nbf.write(notebook, path)
print(f"Notebook executado: {len([c for c in notebook.cells if c.cell_type == 'code'])} celulas de codigo")
