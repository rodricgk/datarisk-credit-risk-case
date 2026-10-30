# Resultados reproduzidos — protocolo temporal corrigido

Gerado por solution.py. Fonte dos valores e metadados: [metrics.json](metrics.json).

## Seleção: novembro/2020 a fevereiro/2021

Somente rótulos disponíveis antes de 01/03/2021 participam da seleção.

| modelo | n | auc | ks | average_precision | brier | prob_media | target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_logistica | 7581 | 0.861859 | 0.596120 | 0.382412 | 0.039046 | 0.056286 | 0.052368 |
| hgb_cfg1 | 7581 | 0.921081 | 0.701417 | 0.490072 | 0.035043 | 0.048816 | 0.052368 |
| hgb_cfg1_weighted | 7581 | 0.917459 | 0.683030 | 0.474785 | 0.060600 | 0.151152 | 0.052368 |
| hgb_cfg1_calibrado | 7581 | 0.918985 | 0.696539 | 0.437493 | 0.036637 | 0.039473 | 0.052368 |
| hgb_cfg2 | 7581 | 0.927219 | 0.722594 | 0.516169 | 0.034162 | 0.043254 | 0.052368 |
| hgb_cfg2_weighted | 7581 | 0.915076 | 0.679893 | 0.459267 | 0.050710 | 0.117347 | 0.052368 |
| hgb_cfg2_calibrado | 7581 | 0.911954 | 0.659113 | 0.446856 | 0.036393 | 0.038948 | 0.052368 |
| hgb_cfg3 | 7581 | 0.920320 | 0.702701 | 0.487826 | 0.035352 | 0.049699 | 0.052368 |
| hgb_cfg3_weighted | 7581 | 0.917246 | 0.700933 | 0.466022 | 0.069734 | 0.174083 | 0.052368 |
| hgb_cfg3_calibrado | 7581 | 0.913205 | 0.683911 | 0.427596 | 0.037229 | 0.039195 | 0.052368 |

## Auditoria: março a junho/2021 — todas as cobranças

Configuração escolhida na janela anterior; histórico congelado em 01/03/2021.

**Ressalva:** este período foi examinado nas versões antigas. Não é um teste prospectivo cego.

| modelo | n | auc | ks | average_precision | brier | prob_media | target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hgb_cfg2 | 9728 | 0.931133 | 0.740890 | 0.585654 | 0.039290 | 0.044057 | 0.063322 |

## Por safra

| modelo | n | auc | ks | average_precision | brier | prob_media | target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2021-03 | 2326 | 0.946952 | 0.806881 | 0.661744 | 0.035635 | 0.058976 | 0.066208 |
| 2021-04 | 2358 | 0.930445 | 0.758385 | 0.550524 | 0.034258 | 0.040184 | 0.053435 |
| 2021-05 | 2531 | 0.930931 | 0.734063 | 0.605237 | 0.047724 | 0.044202 | 0.075464 |
| 2021-06 | 2513 | 0.933169 | 0.724260 | 0.544449 | 0.038899 | 0.033736 | 0.057700 |

## Segmentos

| modelo | n | auc | ks | average_precision | brier | prob_media | target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sem_historico | 176 | 0.961060 | 0.800795 | 0.772860 | 0.060489 | 0.133766 | 0.142045 |
| com_historico | 9552 | 0.930880 | 0.739716 | 0.581584 | 0.038899 | 0.042404 | 0.061872 |
| datas_suspeitas | 7 | 1.000000 | 1.000000 | 1.000000 | 0.039798 | 0.357134 | 0.285714 |
| demais_datas | 9721 | 0.931020 | 0.740523 | 0.583311 | 0.039290 | 0.043832 | 0.063162 |

Os ICs por cliente dos segmentos com duas classes estão no JSON. Grupos pequenos exigem cautela.

## Confiabilidade por decil de score

| faixa | n | predicted | observed |
| --- | --- | --- | --- |
| (-0.000811, 0.000986] | 973 | 0.000749 | 0.000000 |
| (0.000986, 0.00139] | 973 | 0.001178 | 0.005139 |
| (0.00139, 0.00203] | 973 | 0.001680 | 0.002055 |
| (0.00203, 0.00313] | 972 | 0.002534 | 0.005144 |
| (0.00313, 0.00521] | 973 | 0.004083 | 0.003083 |
| (0.00521, 0.00863] | 973 | 0.006707 | 0.009250 |
| (0.00863, 0.0148] | 972 | 0.011476 | 0.022634 |
| (0.0148, 0.0322] | 973 | 0.021931 | 0.034943 |
| (0.0322, 0.1] | 973 | 0.056941 | 0.084275 |
| (0.1, 0.994] | 973 | 0.333216 | 0.466598 |

Brier não isola calibração; as diferenças observado-previsto são diagnósticos descritivos.

## Sensibilidade: incluir datas suspeitas somente no treino

A população de auditoria permanece idêntica; não se usa o resultado para escolher outra configuração.

| modelo | n | auc | ks | average_precision | brier | prob_media | target_medio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| treino_inclui_datas_suspeitas | 9728 | 0.932715 | 0.746807 | 0.593046 | 0.038763 | 0.046821 | 0.063322 |

## Incerteza

Bootstrap por cliente: {'repeats_requested': 300, 'repeats_valid': 300, 'seed': 42, 'auc_95': [0.9076353974401197, 0.9478562801168953], 'brier_95': [0.028774769274065123, 0.05212880876526981]}. Condicional ao modelo; não mede seleção nem regimes futuros.

## População

{'development': 77414, 'suspicious_development': 71, 'selection_total': 9342, 'selection_mature': 7581, 'selection_unmature_excluded': 1761, 'audit': 9728, 'audit_excluded': 0, 'test': 12275}

## Treino final

{'fit_cutoff': '2021-07-01', 'label_cutoff': '2021-07-01', 'train_n': 75540, 'train_latest_emission': '2021-06-15', 'calibration_n': 0}

Sem alegação de ganho financeiro, calibração perfeita ou ausência de todos os vieses.
