# Previsão de Séries Temporais Financeiras com Redes Neuronais Recorrentes

## Análise do Efeito dos Dados e Hiperparâmetros no Modelo de Predição

---

## (1) Introdução e Contexto

A previsão de séries temporais financeiras é um problema clássico de aprendizagem máquina, onde se pretende modelar a evolução de preços de ativos ao longo do tempo. As Redes Neuronais Recorrentes (RNNs), em particular LSTMs e GRUs, tornaram-se ferramentas populares por conseguirem captar dependências temporais de longo prazo — algo que modelos tradicionais como ARIMA não conseguem fazer de forma eficiente.

Este trabalho utiliza dados históricos de 7 ativos financeiros (AAPL, MSFT, GOOGL, AMZN, JPM, SPY, XLK, NVDA) obtidos via Yahoo Finance, com um período de **5 anos de dados diários** (1256 observações, de 2021-05-20 a 2026-05-20). A escolha destes ativos permite analisar o efeito de diferentes correlações com o ativo alvo (AAPL): desde o SPY (mercado total, correlação ~0.75) e XLK (setor Tech, ~0.75) com correlações altas, até ao JPM (setor financeiro, correlação ~0.37) com baixa correlação.

O objetivo principal é avaliar como diferentes escolhas de dados e hiperparâmetros impactam a capacidade preditiva de modelos RNN, utilizando técnicas modernas de validação e avaliação para garantir resultados robustos e generalizáveis.

---

## (2) Definição do Problema

O problema central consiste em prever o preço de fecho diário da Apple (AAPL) para os próximos 7 dias úteis, usando séries temporais de preços históricos como input. Para tal, foram definidas 6 experiências comparativas que isolam diferentes aspetos do modelo:

### Experiências Realizadas

| # | Experiência | Parâmetro Variável | Objetivo |
|---|---|---|---|
| 0 | **Baseline** | — | Estabelecer referência |
| 1 | **Arquiteturas** | LSTM vs BiLSTM vs GRU vs BiGRU | Qual arquitetura é mais adequada? |
| 2 | **Janela Temporal** | n_steps = 3, 5, 10, 20 | Quantos dias de contexto usar? |
| 3 | **Normalizações** | MinMax, StandardScaler, RobustScaler | Qual técnica de scaling funciona melhor? |
| 4 | **Profundidade** | 1-3 camadas, 25-100 unidades | Quantas camadas e unidades? |
| 5 | **Contexto Mercado** | SPY, XLK, OHLC como features extra | Mais features ajudam? |
| 6 | **Grid Search** | Combinações automáticas | Qual a melhor combinação global? |

### Pré-processamento e Validação

Para garantir resultados robustos, foram implementadas as seguintes melhorias relativamente a abordagens anteriores:

1. **Divisão temporal cronológica**: 880 treino (70%), 188 validação (15%), 188 teste (15%) — garantindo que o modelo nunca vê dados futuros durante o treino
2. **Loss de validação** monitorizada durante o treino para detetar overfitting
3. **Métricas no test set**: RMSE, MAE, MAPE e R² calculados em dados nunca vistos
4. **Previsão recursiva multi-passo** no test set (30 dias) para simular condições reais
5. **Grid Search automática** sobre 8 combinações de n_steps, units e layers

---

## (3) Resultados e sua Análise

### Experiência 0 — Baseline

A baseline utiliza uma LSTM simples com 1 camada de 50 unidades, janela de 20 dias, normalização MinMax [0,1] e 50 épocas de treino.

| Métrica | Valor |
|---|---|
| MSE (teste) | 0.000440 |
| RMSE (test set) | $6.20 |
| MAE (test set) | $5.01 |
| MAPE (test set) | 1.90% |
| R² (test set) | 0.842 |

**Análise**: A baseline apresenta um ajuste razoável, com R² de 0.842 no test set, indicando que o modelo consegue explicar 84.2% da variância dos preços. O MAPE de 1.90% é aceitável para previsões financeiras de curto prazo. No entanto, a previsão recursiva de 30 dias revela que o erro tende a acumular-se, com o modelo a desviar-se gradualmente do valor real.

### Experiência 1 — Arquiteturas RNN

Foram testadas 4 arquiteturas com o mesmo número de unidades (50) e 1 camada:

| Arquitetura | Parâmetros | MSE Final |
|---|---|---|
| **LSTM** | ~20.400 | 0.000366 |
| **BiLSTM** | ~40.800 | 0.000428 |
| **GRU** | ~15.300 | 0.000274 |
| **BiGRU** | ~30.600 | **0.000267** |

**Análise**: A BiGRU obteve o melhor MSE (0.000267), superando a LSTM em ~27% e a GRU unidirecional em ~2.6%. Ao contrário das expectativas iniciais (que previam pior desempenho das bidirecionais por usarem contexto futuro), a BiGRU venceu, sugerindo que o processamento bidirecional ajuda a extrair padrões temporais mais robustos mesmo em forecasting. A GRU unidirecional (0.000274) oferece o melhor compromisso desempenho/eficiência, com apenas metade dos parâmetros da BiGRU e um MSE apenas 2.6% superior.

### Experiência 2 — Janela Temporal (n_steps)

Utilizando a BiGRU (melhor arquitetura), variou-se o número de dias de contexto:

| n_steps | MSE Final | Contexto |
|---|---|---|
| 3 | 0.000417 | Muito curto |
| **5** | **0.000339** | **1 semana** |
| 10 | 0.000348 | 2 semanas |
| 20 | 0.000350 | 1 mês |

**Análise**: A janela de 5 dias (1 semana de negociação) apresenta o menor MSE (0.000339), superando 10 dias (0.000348) e 20 dias (0.000350). Com 5 anos de dados disponíveis, a janela curta é suficiente para captar a dinâmica recente do mercado sem introduzir ruído de períodos muito anteriores. A diferença entre 5, 10 e 20 dias é pequena (~3%), indicando robustez à escolha da janela.

### Experiência 3 — Normalizações

Foram testadas 4 técnicas de normalização com a BiGRU e n_steps=5:

| Normalização | MSE Final | Observação |
|---|---|---|
| **MinMax [0,1]** | **0.000377** | Melhor desempenho |
| MinMax [-1,1] | 0.001277 | 3.4× pior |
| StandardScaler | 0.005266 | 14× pior |
| RobustScaler | 0.001957 | 5.2× pior |

**Análise**: O MinMaxScaler [0,1] é claramente superior (MSE=0.000377). O StandardScaler é 14× pior (0.005266) e o RobustScaler 5.2× pior (0.001957) — ambos produzem valores negativos que prejudicam as ativações ReLU. O MinMax [-1,1] (0.001277, 3.4× pior) confirma que o intervalo [0,1] é o mais adequado para RNNs com ReLU.

### Experiência 4 — Profundidade do Modelo

Foram testadas 7 configurações diferentes de camadas e unidades:

| Configuração | MSE Final | Tempo (s) |
|---|---|---|
| 1 camada, [25] | 0.000395 | 18.0 |
| 1 camada, [50] | 0.000358 | 19.5 |
| 1 camada, [100] | 0.000350 | 18.2 |
| 2 camadas, [50, 25] | 0.000331 | 28.2 |
| **2 camadas, [50, 50]** | **0.000327** | **27.5** |
| 3 camadas, [50, 25, 25] | 0.000356 | 32.9 |
| 3 camadas, [50, 50, 25] | 0.000374 | 33.3 |

**Análise**: A configuração com 2 camadas de 50 unidades cada obteve o melhor MSE (0.000327), uma melhoria de ~9% face a 1 camada [50] (0.000358). As 2 camadas permitem aprender representações hierárquicas: a primeira capta padrões de curto prazo, a segunda refina para padrões de mais longo prazo. Com 3 camadas, o MSE piora (0.000356-0.000374), indicando overfitting — mais parâmetros sem dados proporcionais degrada a generalização. O custo de 1 para 2 camadas é moderado (+41% tempo), justificando o ganho.

### Experiência 5 — Contexto de Mercado

Foram testados 5 conjuntos de features diferentes como input com BiGRU e n_steps=5:

| Configuração | Nº Features | MSE Final |
|---|---|---|
| 1D Close (baseline) | 1 | 0.000329 |
| 2D Close + SPY | 2 | 0.000324 |
| 2D Close + XLK | 2 | 0.000362 |
| **Multi OHLC** | **4** | **0.000143** |
| Multi + SPY | 5 | 0.000155 |

**Análise**: A configuração OHLC (Open, High, Low, Close da AAPL) obteve o **melhor MSE global de todo o estudo** (0.000143), uma melhoria de **56.5%** face à configuração 1D (0.000329) e de **67.5%** face à baseline original da Exp 0 (0.000440). As 4 features fornecem informação complementar sobre a dinâmica intradiária que o fecho isolado não capta. O Multi + SPY (0.000155) também é excelente, mas a inclusão do SPY não acrescenta valor quando as 4 features OHLC já estão presentes. Adicionar SPY ou XLK ao Close simples (2 features) dá apenas melhorias marginais (0.000324 e 0.000362, respetivamente).

### Experiência 6 — Grid Search Automático

A grid search testou 8 combinações de n_steps, units e layers com 30 épocas cada:

| Combinação | n_steps | Units | Layers | val_loss |
|---|---|---|---|---|
| 1 | 10 | 50 | 1 | — |
| 2 | 10 | 50 | 2 | — |
| 3 | 10 | 100 | 1 | — |
| 4 | 10 | 100 | 2 | — |
| 5 | 20 | 50 | 1 | — |
| 6 | 20 | 50 | 2 | — |
| **7** | **20** | **100** | **2** | **0.000691** |
| 8 | 20 | 100 | 1 | — |

**Melhores parâmetros**: n_steps=20, units=100, layers=2 (val_loss=0.000691)
**Métricas no test set**: RMSE=$5.64, MAE=$4.75, MAPE=1.79%, R²=0.869

**Análise**: A grid search encontrou n_steps=20, units=100, layers=2 como a melhor combinação (val_loss=0.000691), com R²=0.869 no test set — superior ao baseline (R²=0.842) e às experiências individuais. Apesar de usar apenas 30 épocas (vs 50 nas experiências anteriores), a combinação otimizada generaliza melhor. É de notar que nenhuma experiência individual testou exatamente esta combinação (n_steps=20, 100 units, 2 layers), validando a utilidade da grid search para descobrir configurações não óbvias.

---

## (4) Conclusão

Este estudo demonstrou que a otimização sistemática de dados e hiperparâmetros tem um impacto significativo na performance de modelos RNN para previsão financeira. Partindo de uma baseline LSTM simples (MSE=0.000440), cada experiência contribuiu para reduzir o erro:

1. **BiGRU** (MSE=0.000267) — 39% melhor que a baseline LSTM; a bidirecionalidade melhora a extração de padrões temporais
2. **Janela de 5 dias** (MSE=0.000339) — contexto recente suficiente para 5 anos de dados
3. **MinMax [0,1]** (MSE=0.000377) — essencial para ReLU; StandardScaler é 14× pior
4. **2 camadas [50,50]** (MSE=0.000327) — hierarquia sem overfitting; 3 camadas pioram
5. **OHLC multi-feature** (MSE=**0.000143**) — **67.5% de redução** face à baseline original; a melhor descoberta do estudo
6. **Grid Search** (n_steps=20, units=100, layers=2) — val_loss=0.000691, R²=0.869

A melhoria global — desde a baseline (MSE=0.000440) até ao melhor modelo OHLC (MSE=0.000143) — representa uma redução de erro de **67.5%**, demonstrando que a engenharia de features e a otimização de hiperparâmetros são tão importantes quanto a escolha da arquitetura.

**Limitações e trabalho futuro**: Os resultados devem ser interpretados com cautela, dado que previsões de séries financeiras têm limitações inerentes — os mercados são influenciados por eventos imprevisíveis (notícias, decisões políticas, catástrofes). A implementação de **validação walk-forward** completa, a incorporação de **indicadores técnicos** (RSI, MACD) e **dados macroeconómicos** como features adicionais, e o teste de **modelos Transformer** seriam os passos naturais seguintes.

---

*Anilson Monteiro — Pós-Graduação em Ciência de Dados Aplicada à Análise de Risco*
*Maio 2026*
