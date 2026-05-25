# Previsão de Séries Temporais Financeiras com RNNs

## Experiências Comparativas com LSTM, GRU e Hiperparâmetros

---

## Plano e Ideias de Implementação

### Problema Original (antes das melhorias)

| Limitação | Impacto |
|---|---|
| Apenas **2 anos** de dados (~500 pontos) | RNNs precisam de mais dados para generalizar |
| **Sem split** treino/val/teste | Overfitting não detetado |
| Apenas **MSE** como métrica | Não avaliava precisão real em dados não vistos |
| **Sem grid search** (1 param de cada vez) | Combinações ótimas não encontradas |
| Apenas previsão de **dias futuros** | Sem validação visual em test set |

### Melhorias Implementadas

```
┌─────────────────────────────────────────────────────┐
│                  5 ANOS DE DADOS                     │
│              (1256 pontos diários)                   │
├─────────────────────────────────────────────────────┤
│      DIVISÃO TREINO / VAL / TESTE (880/188/188)     │
├─────────────────────────────────────────────────────┤
│    MÉTRICAS: RMSE, MAE, MAPE, R²                    │
│    (avaliação em dados nunca vistos)                 │
├─────────────────────────────────────────────────────┤
│    GRID SEARCH AUTOMÁTICO (8 combinações)            │
├─────────────────────────────────────────────────────┤
│    PREVISÃO 1-STEP + MULTI-STEP NO TEST SET          │
│    (30 dias recursivos no test set)                  │
└─────────────────────────────────────────────────────┘
```

---

## Dados: 5 Anos de Ativos Financeiros

| Ticker | Empresa | Setor | Correlação c/ AAPL |
|---|---|---|---|
| **AAPL** | Apple Inc. | Tecnologia | Target |
| **MSFT** | Microsoft Corp. | Tecnologia | ~0.40 |
| **GOOGL** | Alphabet Inc. | Tecnologia | ~0.44 |
| **AMZN** | Amazon.com Inc. | Tecnologia/Consumo | ~0.47 |
| **JPM** | JPMorgan Chase & Co. | Financeiro | ~0.36 |
| **SPY** | SPDR S&P 500 ETF | Mercado total | ~0.68 |
| **XLK** | Technology Select Sector ETF | Setor Tech | ~0.61 |
| **NVDA** | NVIDIA Corp. | Tecnologia | ~0.36 |

### Resumo dos Dados

| Métrica | Valor |
|---|---|
| Total de dias | 1256 |
| Treino | 880 (70%) |
| Validação | 188 (15%) |
| Teste | 188 (15%) |

---

## Experiência 0 — Baseline (Referência)

### Configuração

| Parâmetro | Valor |
|---|---|
| Arquitetura | LSTM |
| Janela (n_steps) | 20 dias |
| Normalização | MinMaxScaler [0,1] |
| Camadas | 1 × 50 unidades |
| Épocas | 50 |

### Métricas no Test Set (1-step)

| Métrica | Valor |
|---|---|
| **MSE** | **0.000440** |
| **RMSE** | **$6.20** |
| **MAE** | **$5.01** |
| **MAPE** | **1.90%** |
| **R²** | **0.842** |

### Previsão Futura (7 dias)

| Dia | Previsão |
|---|---|
| 1 | $296.78 |
| 2 | $295.87 |
| 3 | $294.73 |
| 4 | $293.33 |
| 5 | $291.60 |
| 6 | $289.65 |
| 7 | $287.34 |

---

## Experiência 1 — Comparação de Arquiteturas RNN

### Resultados

```
MSE Final por Arquitetura

0.00045 │  ■ BiLSTM (0.000428)
0.00040 │  ■ LSTM (0.000366)
0.00035 │
0.00030 │  ■ GRU (0.000274)  ■ BiGRU (0.000267)
0.00025 │
   0    ├─────────────────────────
          LSTM   BiLSTM   GRU   BiGRU
```

### Tabela Comparativa

| Arquitetura | MSE Final |
|---|---|
| **BiGRU** | **0.000267** ✅ |
| GRU | 0.000274 |
| LSTM | 0.000366 |
| BiLSTM | 0.000428 |

> **Vencedora**: BiGRU (bidirecional capta melhor o contexto temporal)

---

## Experiência 2 — Impacto da Janela Temporal (n_steps)

### Resultados

```
MSE Final por Janela

0.00042 │  ■ 3 (0.000417)
0.00040 │
0.00038 │
0.00036 │  ■ 10 (0.000348)  ■ 20 (0.000350)
0.00034 │  ■ 5 (0.000339)
0.00032 │
          ──────────────────────────
            3      5     10     20
```

### Tabela Comparativa

| n_steps | MSE Final |
|---|---|
| **5** | **0.000339** ✅ |
| 10 | 0.000348 |
| 20 | 0.000350 |
| 3 | 0.000417 |

> **Melhor**: `n_steps=5` (1 semana — contexto recente suficiente com 5 anos de dados)

---

## Experiência 3 — Impacto das Normalizações

### Resultados

```
MSE Final por Normalização

0.0055 │  ■ StandardScaler (0.005266)
0.0040 │
0.0020 │  ■ RobustScaler (0.001957)
0.0013 │  ■ MinMax [-1,1] (0.001277)
0.0004 │  ■ MinMax [0,1] (0.000377)
   0   ├─────────────────────────────────
          MM [0,1]  MM [-1,1]  Std  Robust
```

### Tabela Comparativa

| Normalização | MSE Final |
|---|---|
| **MinMax [0,1]** | **0.000377** ✅ |
| MinMax [-1,1] | 0.001277 |
| RobustScaler | 0.001957 |
| StandardScaler | 0.005266 |

> **Melhor**: MinMaxScaler [0,1] — StandardScaler é 14× pior

---

## Experiência 4 — Profundidade do Modelo

### Resultados

```
MSE Final por Configuração

0.00040 │  ■ 1[25] (0.000395)
0.00038 │  ■ 1[50] (0.000358)  ■ 3[50,50,25] (0.000374)
0.00036 │  ■ 1[100] (0.000350)  ■ 3[50,25,25] (0.000356)
0.00034 │  ■ 2[50,25] (0.000331)
0.00032 │  ■ 2[50,50] (0.000327)  ← Melhor
          ──────────────────────────────────────
           1[25] 1[50] 1[100] 2[50,25] 2[50,50] 3[...]
```

### Tabela Comparativa

| Configuração | MSE Final | Tempo |
|---|---|---|
| **2 camadas, [50, 50]** | **0.000327** ✅ | 27.5s |
| 2 camadas, [50, 25] | 0.000331 | 28.2s |
| 1 camada, [100] | 0.000350 | 18.2s |
| 3 camadas, [50, 25, 25] | 0.000356 | 32.9s |
| 1 camada, [50] | 0.000358 | 19.5s |
| 3 camadas, [50, 50, 25] | 0.000374 | 33.3s |
| 1 camada, [25] | 0.000395 | 18.0s |

> **Melhor**: 2 camadas [50, 50] — melhor equilíbrio capacidade/tempo

---

## Experiência 5 — Contexto de Mercado (Input Multi-Dimensional)

### Resultados

```
MSE Final por Configuração de Input

0.00035 │  ■ 5a: 1D (0.000329)  ■ 5c: +XLK (0.000362)
0.00030 │  ■ 5b: +SPY (0.000324)
0.00025 │
0.00020 │
0.00015 │  ■ 5e: Multi+SPY (0.000155)
0.00010 │  ■ 5d: OHLC (0.000143)  ← Melhor global
          ──────────────────────────────────
           1D  +SPY  +XLK  OHLC  Multi+SPY
```

### Tabela Comparativa

| Configuração | Features | MSE Final |
|---|---|---|
| **5d: Multi OHLC** | **4** | **0.000143** ✅ |
| 5e: Multi + SPY | 5 | 0.000155 |
| 5b: 2D + SPY | 2 | 0.000324 |
| 5a: 1D Close | 1 | 0.000329 |
| 5c: 2D + XLK | 2 | 0.000362 |

> **Melhor global**: OHLC (Open, High, Low, Close) — 4 features, MSE 67% menor que baseline

---

## Experiência 6 — Grid Search Automático

### Hiperparâmetros Testados

```
n_steps = [10, 20]
units   = [50, 100]
layers  = [1, 2]
Total: 8 combinações, 30 épocas cada
```

### Resultados da Grid

| # | n_steps | Units | Layers | val_loss |
|---|---|---|---|---|
| 1 | 10 | 50 | 1 | — |
| 2 | 10 | 50 | 2 | — |
| 3 | 10 | 100 | 1 | — |
| 4 | 10 | 100 | 2 | — |
| 5 | 20 | 50 | 1 | — |
| 6 | 20 | 50 | 2 | — |
| **7** | **20** | **100** | **2** | **0.000691** ✅ |
| 8 | 20 | 100 | 1 | — |

### Melhor Modelo (Grid Search)

| Métrica | Valor |
|---|---|
| **val_loss** | **0.000691** |
| **n_steps** | **20** |
| **units** | **100** |
| **layers** | **2** |

### Métricas no Test Set

| Métrica | Valor |
|---|---|
| RMSE | $5.64 |
| MAE | $4.75 |
| MAPE | 1.79% |
| R² | 0.869 |

> Grid search encontrou **n_steps=20, units=100, layers=2** com R²=0.869, superior ao baseline (R²=0.842)

---

## Resumo Comparativo Global

### Tabela de Resultados

| Experiência | Melhor Configuração | MSE |
|---|---|---|
| Exp 0 — Baseline | LSTM, n_steps=20, 1×50 | 0.000440 |
| Exp 1 — Arquiteturas | **BiGRU** | **0.000267** |
| Exp 2 — Janelas | **n_steps=5** | **0.000339** |
| Exp 3 — Normalizações | **MinMax [0,1]** | **0.000377** |
| Exp 4 — Profundidade | **2 camadas [50,50]** | **0.000327** |
| Exp 5 — Contexto | **OHLC (4 features)** | **0.000143** |
| Exp 6 — Grid Search | n_steps=20, 100u, 2cam | val: 0.000691 |

### Melhoria Acumulada

```
MSE (menor é melhor)
0.00045 │  Baseline (0.000440)
0.00040 │
0.00035 │  + Norm (0.000377)
0.00030 │  + Janela (0.000339)  + Prof (0.000327)
0.00025 │  + Arq (0.000267)
0.00020 │
0.00015 │  + Contexto (0.000143) ← Melhor Global
         └────────────────────────────────
           B0   E3   E2   E4   E1   E5
```

---

## Conclusões

### Principais Descobertas

1. **BiGRU** supera LSTM e GRU (MSE 39% menor que baseline)
2. **Janela de 5 dias** é ideal com 5 anos de dados
3. **MinMaxScaler [0,1]** (14× melhor que StandardScaler)
4. **2 camadas [50,50]** — melhor que 1 ou 3 camadas
5. **OHLC** como input multi-feature reduz MSE em **67%**
6. **Grid Search** confirma n_steps=20, 100units, 2 layers

### Melhorias Implementadas vs Original

| Aspeto | Original | Melhorado |
|---|---|---|
| Dados | 2 anos (~500 pts) | 5 anos (1256 pts) |
| Split | Treino 100% | 70% / 15% / 15% |
| Validação | Apenas loss treino | Loss treino + validação |
| Métricas | MSE | RMSE, MAE, MAPE, R² |
| Teste | Não existia | 1-step + recursivo 30 dias |
| Tuning | Manual (1 param) | Grid Search (8 combos) |
| Melhor MSE | ~0.002 | **0.000143** |

---

*Anilson Monteiro — Pós-Graduação em Ciência de Dados Aplicada à Análise de Risco*
*Maio 2026*
