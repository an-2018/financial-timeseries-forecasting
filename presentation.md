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
│              (~1250 pontos diários)                  │
├─────────────────────────────────────────────────────┤
│         DIVISÃO TREINO / VAL / TESTE                 │
│             (70% / 15% / 15%)                        │
├─────────────────────────────────────────────────────┤
│    MÉTRICAS: RMSE, MAE, MAPE, R²                    │
│    (avaliação em dados nunca vistos)                 │
├─────────────────────────────────────────────────────┤
│    GRID SEARCH AUTOMÁTICO                            │
│    (n_steps, units, layers combinados)               │
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

### Evolução dos Preços (base 100)

```
Preço
 600 │        NVDA
 500 │          ▲
 400 │        /
 300 │  AAPL
 200 │ /│  MSFT
 100 ├───────────────────────►
     2021  2022  2023  2024  2025  2026
          5 anos de dados
```

### Matriz de Correlação (Retornos Logarítmicos)

```
        AAPL  MSFT GOOGL  AMZN   JPM   SPY   XLK   NVDA
AAPL    1.00  0.40  0.44  0.47  0.36  0.68  0.61  0.36
MSFT    0.40  1.00  0.53  0.49  0.34  0.65  0.72  0.39
GOOGL   0.44  0.53  1.00  0.48  0.32  0.61  0.62  0.37
AMZN    0.47  0.49  0.48  1.00  0.33  0.62  0.57  0.41
JPM     0.36  0.34  0.32  0.33  1.00  0.53  0.39  0.28
SPY     0.68  0.65  0.61  0.62  0.53  1.00  0.76  0.56
XLK     0.61  0.72  0.62  0.57  0.39  0.76  1.00  0.51
NVDA    0.36  0.39  0.37  0.41  0.28  0.56  0.51  1.00
```

> **Observação**: SPY (mercado total) e XLK (setor Tech) têm as correlações mais altas com AAPL. JPM (Financeiro) tem a correlação mais baixa.

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
| Split | 70% Treino / 15% Val / 15% Teste |

### Loss Curve (Treino + Validação)

```
MSE
0.10 │
0.08 │        ╱╲
0.06 │      ╱   ╲___ Treino
0.04 │    ╱         ╲_ Validação
0.02 │  ╱
0.00 ├────────────────────►
      0   10   20   30   40   50
                 Épocas
```

### Métricas no Test Set (1-step)

| Métrica | Valor |
|---|---|
| **RMSE** | ~$4.50 |
| **MAE** | ~$3.20 |
| **MAPE** | ~1.8% |
| **R²** | ~0.85 |

### Previsão 1-Step no Test Set

```
Preço ($)
  250 │              ╱╲    ╱╲
      │    ╱╲  ╱╲  ╱  ╲  ╱  ╲
  240 │  ╱  ╲╱  ╲╱    ╲╱    ╲__ Real
      │ ╱                          __ Previsto
  230 ├────────────────────────────────►
      │         Dias no Test Set
```

### Previsão Recursiva (30 dias) no Test Set

```
Preço ($)
    │  Treino    │ Val  │  Teste
  250│ ╱╲  ╱╲    │  ╱╲  │ ╱╲    ╱╲
    │╱  ╲╱  ╲   │ ╱  ╲ │╱  ╲  ╱  ╲
  240│         ╲ │╱    ╲│    ╲╱    ╲__ Real
    │          ╲│      ╲│          ╲__ Previsto
  230├────────────────────────────────►
    │         Datas
```

### Previsão Futura (7 dias)

| Dia | Previsão |
|---|---|
| 1 | $303.05 |
| 2 | $304.12 |
| 3 | $304.89 |
| 4 | $305.45 |
| 5 | $305.78 |
| 6 | $305.92 |
| 7 | $306.01 |

---

## Experiência 1 — Comparação de Arquiteturas RNN

### Arquiteturas Testadas

| Arquitetura | Parâmetros | Bidirecional | Portas |
|---|---|---|---|
| **LSTM** | ~20.400 | Não | 3 |
| **BiLSTM** | ~40.800 | Sim | 3 |
| **GRU** | ~15.300 | Não | 2 |
| **BiGRU** | ~30.600 | Sim | 2 |

### Resultados

```
MSE Final por Arquitetura

0.0025 │  ■ BiLSTM (0.00231)
0.0020 │  ■ LSTM (0.00173)
0.0015 │  ■ BiGRU (0.00131)
0.0010 │  ■ GRU (0.00121)
0.0005 │
  0    ├─────────────────────
          LSTM  BiLSTM  GRU  BiGRU
```

### Tabela Comparativa

| Arquitetura | MSE Final |
|---|---|
| **GRU** | **0.001211** ✅ |
| BiGRU | 0.001306 |
| LSTM | 0.001726 |
| BiLSTM | 0.002312 |

> **Vencedora**: GRU (menor MSE, menos parâmetros, treino mais rápido)

---

## Experiência 2 — Impacto da Janela Temporal (n_steps)

### Configurações Testadas

| Janela | Dias Úteis | Contexto |
|---|---|---|
| 3 | ~meia semana | Curto prazo |
| 5 | 1 semana | Semana de negociação |
| 10 | 2 semanas | Médio prazo |
| 20 | 1 mês | Longo prazo |

### Resultados

```
MSE Final por Janela

0.0014 │  ■ 3 (0.00138)
0.0013 │  ■ 5 (0.00136)
0.0012 │  ■ 20 (0.00116)
0.0011 │  ■ 10 (0.00110)
0.0010 │
        ────────────────────────
          3    5    10    20
```

### Tabela Comparativa

| n_steps | MSE Final |
|---|---|
| 10 | **0.001098** ✅ |
| 20 | 0.001159 |
| 5 | 0.001362 |
| 3 | 0.001384 |

> **Melhor**: `n_steps=10` (equilíbrio ideal entre contexto e ruído)

---

## Experiência 3 — Impacto das Normalizações

### Técnicas Testadas

| Scaler | Fórmula | Range |
|---|---|---|
| MinMax [0,1] | (x - min) / (max - min) | [0, 1] |
| MinMax [-1,1] | 2×(x-min)/(max-min) - 1 | [-1, 1] |
| StandardScaler | (x - μ) / σ | ~[-3, 3] |
| RobustScaler | (x - mediana) / IQR | ~[-3, 3] |

### Resultados

```
MSE Final por Normalização

0.025 │  ■ StandardScaler (0.0223)
0.020 │
0.015 │
0.010 │  ■ RobustScaler (0.0089)
0.005 │  ■ MinMax [-1,1] (0.0036)
  0    │  ■ MinMax [0,1] (0.0011)
       ──────────────────────────────────
         MM [0,1]  MM [-1,1]  Std  Robust
```

### Tabela Comparativa

| Normalização | MSE Final |
|---|---|
| **MinMax [0,1]** | **0.001106** ✅ |
| MinMax [-1,1] | 0.003558 |
| RobustScaler | 0.008851 |
| StandardScaler | 0.022332 |

> **Melhor**: MinMaxScaler [0,1] — preserva a forma da distribuição e é o mais adequado para séries financeiras

---

## Experiência 4 — Profundidade do Modelo

### Configurações Testadas

| Nº Camadas | Unidades | Tempo (s) |
|---|---|---|
| 1 | [25] | ~12s |
| 1 | [50] | ~16s |
| 1 | [100] | ~18s |
| 2 | [50, 25] | ~22s |
| 2 | [50, 50] | ~26s |
| 3 | [50, 25, 25] | ~34s |
| 3 | [50, 50, 25] | ~40s |

### Resultados

```
MSE Final por Configuração

0.0015 │  ■ 3 cam [50,25,25] (0.00147)
0.0014 │
0.0013 │
0.0012 │  ■ 1[25]  1[50]  1[100]
0.0011 │  ■ 2[50,25]  2[50,50]  ← Melhor
0.0010 │
        ──────────────────────────────
         1[25] 1[50] 1[100] 2[50,25] 2[50,50] 3[50,25,25]
```

### Tabela Comparativa

| Configuração | MSE Final | Tempo |
|---|---|---|
| 2 camadas, [50, 50] | **0.001059** ✅ | 26.3s |
| 1 camada, [100] | 0.001125 | 18.4s |
| 2 camadas, [50, 25] | 0.001140 | 22.1s |
| 1 camada, [25] | 0.001170 | 12.1s |
| 1 camada, [50] | 0.001191 | 16.1s |
| 3 camadas, [50, 25, 25] | 0.001474 | 34.0s |

> **Melhor**: **2 camadas [50, 50]** — hierarquia sem overfitting

---

## Experiência 5 — Contexto de Mercado (Input Multi-Dimensional)

### Configurações Testadas

| Variante | Features | Shape |
|---|---|---|
| 5a: 1D Close | Close AAPL | (n, steps, 1) |
| 5b: 2D + SPY | Close AAPL + Close SPY | (n, steps, 2) |
| 5c: 2D + XLK | Close AAPL + Close XLK | (n, steps, 2) |
| 5d: Multi OHLC | Open, High, Low, Close AAPL | (n, steps, 4) |
| 5e: Multi + SPY | OHLC AAPL + Close SPY | (n, steps, 5) |

### Resultados

```
MSE Final por Configuração de Input

0.003 │  ■ 5e: Multi+SPY (0.00285)
0.002 │  ■ 5d: OHLC (0.00212)
0.001 │  ■ 5a: 1D 5b: +SPY 5c: +XLK
      │    (0.0014-0.0015)
  0   ├───────────────────────────────
         1D  +SPY  +XLK  OHLC  Multi+SPY
```

### Tabela Comparativa

| Configuração | Features | MSE Final |
|---|---|---|
| 5b: 2D + SPY | 2 | **0.001391** ✅ |
| 5c: 2D + XLK | 2 | 0.001422 |
| 5a: 1D Close | 1 | 0.001498 |
| 5d: Multi OHLC | 4 | 0.002118 |
| 5e: Multi + SPY | 5 | 0.002851 |

> **Melhor**: Close + SPY (2 features) — contexto de mercado global ajuda, mas mais features podem diluir o sinal

---

## Experiência 6 — Grid Search Automático

### Hiperparâmetros Testados

```
n_steps = [10, 20]
units   = [50, 100]
layers  = [1, 2]
─────────────────────
Total: 8 combinações
```

### Resultados da Grid

```
Progresso da Grid Search
[1/8] n_steps=10, units=50,  layers=1  → val_loss=0.0032
[2/8] n_steps=10, units=50,  layers=2  → val_loss=0.0028
[3/8] n_steps=10, units=100, layers=1  → val_loss=0.0021
[4/8] n_steps=10, units=100, layers=2  → val_loss=0.0019
[5/8] n_steps=20, units=50,  layers=1  → val_loss=0.0025
[6/8] n_steps=20, units=50,  layers=2  → val_loss=0.0023
[7/8] n_steps=20, units=100, layers=1  → val_loss=0.0017  ← Melhor
[8/8] n_steps=20, units=100, layers=2  → val_loss=0.0018
```

### Melhores Parâmetros

| Parâmetro | Valor |
|---|---|
| **n_steps** | **20** |
| **units** | **100** |
| **layers** | **1** |
| **val_loss** | **0.0017** |

### Métricas no Test Set (Grid Search)

| Métrica | Valor |
|---|---|
| RMSE | ~$3.80 |
| MAE | ~$2.90 |
| MAPE | ~1.5% |
| R² | ~0.89 |

> A grid search encontrou uma combinação superior (n_steps=20, units=100, layers=1) que não tinha sido testada nas experiências individuais

---

## Resumo Comparativo Global

### Tabela de Resultados

| Experiência | Melhor Configuração | MSE |
|---|---|---|
| Exp 0 — Baseline | LSTM, n_steps=20, 1×50 | 0.002023 |
| Exp 1 — Arquiteturas | **GRU** | **0.001211** |
| Exp 2 — Janelas | **n_steps=10** | **0.001098** |
| Exp 3 — Normalizações | **MinMax [0,1]** | **0.001106** |
| Exp 4 — Profundidade | **2 camadas [50,50]** | **0.001059** |
| Exp 5 — Contexto | **Close + SPY** | **0.001391** |
| **Exp 6 — Grid Search** | **n_steps=20, 100u, 1cam** | **val: 0.0017** |

### Ranking de Melhorias

```
MSE (menor é melhor)
0.0020 │  Baseline
0.0018 │
0.0016 │
0.0014 │  + Contexto   + Norm
0.0012 │  + Arq  + Janela
0.0010 │  + Profundidade
       └─────────────────────────
         B0   E1   E2   E3   E4   E5
```

---

## Conclusões

### Principais Descobertas

1. **GRU > LSTM** para forecasting financeiro
2. **Janela de 10 dias** é o equilíbrio ideal
3. **MinMaxScaler [0,1]** é a melhor normalização
4. **2 camadas** melhor que 1 ou 3
5. **SPY como feature extra** melhora previsões
6. **Grid Search** encontra combinações superiores

### Melhorias Implementadas vs Original

| Aspeto | Original | Melhorado |
|---|---|---|
| Dados | 2 anos (~500 pts) | 5 anos (~1250 pts) |
| Split | Treino 100% | Treino 70% / Val 15% / Teste 15% |
| Validação | Apenas loss treino | Loss treino + validação |
| Métricas | MSE | RMSE, MAE, MAPE, R² |
| Teste | Não existia | 1-step + recursivo 30 dias |
| Tuning | Manual (1 param) | Grid Search (8 combos) |

---

*Anilson Monteiro — Pós-Graduação em Ciência de Dados Aplicada à Análise de Risco*
*Maio 2026*
