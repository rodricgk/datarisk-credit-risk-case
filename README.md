# Previsão de Inadimplência — Case Datarisk

Projeto de portfólio baseado no [case público de Ciência de Dados Júnior da Datarisk](https://github.com/datarisk-io/datarisk-case-ds-junior). Estima a probabilidade de uma **cobrança** ser paga com pelo menos cinco dias de atraso.

O foco desta versão é a validade do experimento: histórico disponível no momento correto, regras consistentes entre treino e aplicação e métricas com população e limitações explícitas.

## Resultados e evidências

Os resultados atualizados estão em **[Resultados reproduzidos](reports/metrics.md)**.
A fonte única é [metrics.json](reports/metrics.json), com seleção, auditoria, safras, segmentos, confiabilidade por decil, intervalos por cliente, versões e hashes dos arquivos.

O [notebook executado](notebooks/data_quality_audit.ipynb) mostra os testes nas bases completas e executa a grade estendida. Os números da versão anterior (incluindo AUC 0,9403) foram substituídos: o protocolo, a população e a informação disponível mudaram. Não interpretar a diferença como efeito isolado do algoritmo.

## Problema e alvo

`INADIMPLENTE = 1` quando `DATA_PAGAMENTO - DATA_VENCIMENTO >= 5 dias`.

A unidade é a cobrança, não o cliente. Um cliente pode ter várias cobranças; por isso, os intervalos reamostram clientes completos. Não se deduplicam cobranças apenas por cliente e safra.

As probabilidades podem subsidiar priorização de cobrança. Não foi demonstrado ganho financeiro, efeito causal de uma intervenção ou adequação para concessão automática de crédito.

## Protocolo temporal

Hipótese operacional: prever na emissão da cobrança, usando atributos da cobrança naquele momento e um histórico comportamental mensal. Datas não têm hora; eventos no próprio dia do corte são tratados como ainda desconhecidos.

| Etapa | Janela | Regra |
|---|---|---|
| Escolha da configuração | novembro/2020–fevereiro/2021 | Treino antes de 01/11/2020; comparar somente alvos conhecidos antes de 01/03/2021 |
| Auditoria da configuração escolhida | março–junho/2021 | Novo ajuste antes de 01/03/2021; avaliar todas as 9.728 cobranças, sem filtro por pagamento |
| Aplicação final | julho–novembro/2021 | Novo ajuste antes de 01/07/2021; preservar todas as 12.275 linhas e a ordem original |

**Março–junho já foi analisado nas versões anteriores.** A separação atual evita selecionar hiperparâmetros nessa janela dentro desta execução, mas não transforma um período já examinado em teste prospectivo cego. Ainda é necessário um período novo para confirmação.

### O que pode ser conhecido no corte?

- Cobranças anteriores: somente emissões estritamente anteriores ao corte.
- Alvo binário: conhecido após pagamento antes do limiar, ou após vencimento + 5 dias sem pagamento anterior.
- Atraso final: conhecido somente após observar o pagamento.
- Contagem de cobranças: inclui pendentes; taxa de inadimplência usa como denominador apenas os rótulos já conhecidos.
- Valor médio: média dos valores não nulos, com a mesma função no treino e na aplicação.
- Histórico de cada linha do treino: congelado no início de sua safra.
- Histórico da seleção, auditoria e teste: congelado no início de cada bloco, sem incorporar pagamentos posteriores durante esse bloco.

O treino utiliza apenas rótulos maduros no corte. A seleção também exige rótulos maduros antes da auditoria; o relatório informa quantas cobranças ainda não eram elegíveis. Essa seleção por maturação pode mudar a composição da amostra e é uma limitação explícita.

### Cadastro e informação mensal

O cadastro é tratado como estático e a informação mensal como disponível na emissão. **Isso é uma hipótese, não uma garantia verificada:** as bases não fornecem versões cadastrais ou horários de ingestão. O controle temporal implementado cobre eventos de cobrança/pagamento; não permite afirmar ausência de todo vazamento possível na fonte.

## Qualidade e features

- Contratos verificam colunas, chaves não nulas, unicidade das dimensões, formato de safra, tipos de datas e cardinalidade `many_to_one` dos joins.
- Cadastro ausente recebe `DESCONHECIDO`; não é confundido com PJ cadastrado cujo `FLAG_PF` está vazio.
- Ausência de linha mensal é separada de renda ou quantidade de funcionários não informadas.
- DDDs inválidos são normalizados como desconhecidos.
- Cliente sem histórico tem contagens zero, médias desconhecidas e uma flag que reconhece zero e ausência. Essa representação é compartilhada pelo treino e pelo scoring.
- A última safra sem rótulo conhecido não é convertida artificialmente em adimplência.

### Datas suspeitas e população

Há 71 cobranças de desenvolvimento com prazo negativo, prazo superior a 400 dias ou pagamento anterior à emissão registrada. São **anomalias sob uma regra conservadora**, não erros comprovados: antecipação, renegociação ou reemissão podem explicar parte delas.

Por padrão, seus rótulos não entram no ajuste ou nos agregados de resultados. A existência da cobrança e seu valor continuam compondo o histórico quando observáveis. Nunca se filtra o CSV inteiro antes de construir os históricos.

A auditoria principal inclui todas as cobranças, inclusive suspeitas. O relatório mostra esse segmento separadamente e testa a sensibilidade de incluir esses rótulos no ajuste, mantendo fixa a população de auditoria. O teste não perde linhas.

Flags constantes no treino, como certos indicadores de ausência, não garantem que o modelo aprenda como tratar situações novas. São sinalização, não solução automática para mudança de distribuição.

## Modelos e avaliação

Grade padrão: regressão logística sem balanceamento e três configurações de `HistGradientBoostingClassifier`. Seleção pelo menor Brier, com AUC como desempate. Early stopping aleatório desativado.

O modo `--extended` acrescenta pesos balanceados e calibração isotônica. O vencedor **pode mudar**; não se presume que calibrar ou balancear melhora ou piora o resultado.

Calibração usa os três meses completos anteriores ao corte de ajuste. O modelo-base e todo o pré-processamento são ajustados antes dessa janela e congelados com `FrozenEstimator`. Os rótulos de calibração devem estar maduros no corte externo. A mesma rotina é usada na seleção, auditoria e aplicação final; não há divisão temporal por posição de linhas.

São reportados AUC, Gini, KS com tratamento de empates, Average Precision, Brier e Log Loss. Brier Skill usa como referência a previsão constante igual à prevalência **da amostra avaliada**: é referência descritiva retrospectiva, não uma baseline operacional disponível antecipadamente.

A confiabilidade das probabilidades é examinada por faixas de score, safra e presença de histórico. Brier combina aspectos de discriminação e calibração; sua variação sozinha não comprova mudança de calibração.

Os intervalos usam 300 reamostragens por cliente, semente 42, e são condicionais ao modelo ajustado. Não incorporam incerteza da seleção, reestimação do modelo ou futuros regimes temporais.

## Como reproduzir

Python **3.11 ou superior**; execução validada com Python 3.14. Dependências fixadas em [requirements.txt](requirements.txt).

1. Instale: `pip install -r requirements.txt`.
2. Baixe os quatro CSVs da pasta [data do case oficial](https://github.com/datarisk-io/datarisk-case-ds-junior/tree/master/data). Não são republicados neste repositório.
3. Execute:

```bash
python -m unittest discover -s tests -v
python solution.py --data-dir /caminho/das/bases
# Reproduz a grade completa usada no notebook e no relatório publicado:
python solution.py --extended --data-dir /caminho/das/bases
```

Arquivos esperados: `base_cadastral.csv`, `base_info.csv`, `base_pagamentos_desenvolvimento.csv` e `base_pagamentos_teste.csv`.

O comando gera `submissao_case.csv`, `reports/metrics.json` e `reports/metrics.md`.
Use `--output` e `--report-dir` para manter execuções separadas. Rodar a grade padrão substitui o relatório pela execução padrão; o campo `extended` registra qual grade foi usada.

Para executar o notebook ou reconstruí-lo, defina `DATARISK_DATA_DIR` no ambiente e execute `python scripts/build_audit_notebook.py`. Esse comando recria o notebook a partir do script e executa todas as células. Não é necessário rodar o CLI antes dele.

## Testes e limitações restantes

Os testes cobrem fronteira do target, eventos no corte, maturação de rótulos, invariância a pagamentos futuros, paridade treino/aplicação, nulos históricos, cadastro desconhecido, preservação da ordem, população de avaliação e isolamento da calibração.

Limitações que código sozinho não resolve:

- disponibilidade histórica real de cadastro e informação mensal;
- confiabilidade e cobertura dos desfechos fornecidos pelo case;
- auditoria retrospectiva em um período já examinado;
- grupos pequenos e mudanças de distribuição;
- diferença entre um snapshot fixo e uma operação que atualiza o histórico mensalmente.

Próximo passo científico: validar em um período novo, com datas de disponibilidade registradas e critérios de negócio definidos previamente.

## Fonte e contexto

Projeto independente de portfólio baseado no [case oficial](https://github.com/datarisk-io/datarisk-case-ds-junior). A Datarisk não participou da implementação e não endossa a solução. Bases e submissão não são versionadas.

