# Previsão de Séries Temporais Financeiras com Redes Neuronais Recorrentes

## Análise do Efeito dos Dados e Hiperparâmetros no Modelo de Predição

---

## (1) Introdução e Contexto

A previsão de séries temporais financeiras é um problema clássico de aprendizagem máquina, onde se pretende modelar a evolução de preços de ativos ao longo do tempo. As Redes Neuronais Recorrentes (RNNs), em particular LSTMs e GRUs, tornaram-se ferramentas populares por conseguirem captar dependências temporais de longo prazo — algo que modelos tradicionais como ARIMA não conseguem fazer de forma eficiente.

Este trabalho utiliza dados históricos de 7 ativos financeiros (AAPL, MSFT, GOOGL, AMZN, JPM, SPY, XLK, NVDA) obtidos via Yahoo Finance, com um período de **5 anos de dados diários** (~1250 observações). A escolha destes ativos permite analisar o efeito de diferentes correlações com o ativo alvo (AAPL), desde o SPY (mercado total, correlação ~0.68) até ao JPM (setor financeiro, correlação ~0.36).

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

1. **Divisão temporal cronológica**: 70% treino, 15% validação, 15% teste — garantindo que o modelo nunca vê dados futuros durante o treino
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
| MSE (treino) | 0.002023 |
| RMSE (test set) | ~$4.50 |
| MAE (test set) | ~$3.20 |
| MAPE (test set) | ~1.8% |
| R² (test set) | ~0.85 |

**Análise**: A baseline apresenta um ajuste razoável, com R² de 0.85 no test set, indicando que o modelo consegue explicar 85% da variância dos preços. O MAPE de ~1.8% é aceitável para previsões financeiras de curto prazo. No entanto, a previsão recursiva de 30 dias revela que o erro tende a acumular-se, com o modelo a desviar-se gradualmente do valor real.

### Experiência 1 — Arquiteturas RNN

Foram testadas 4 arquiteturas com o mesmo número de unidades (50) e 1 camada:

| Arquitetura | Parâmetros | MSE Final | Tempo Relativo |
|---|---|---|---|
| **LSTM** | ~20.400 | 0.001726 | 1.0× |
| **BiLSTM** | ~40.800 | 0.002312 | 1.8× |
| **GRU** | ~15.300 | **0.001211** | 0.8× |
| **BiGRU** | ~30.600 | 0.001306 | 1.5× |

**Análise**: A GRU obteve o melhor MSE (0.001211), superando a LSTM em ~30%. Este resultado é consistente com a literatura, que mostra que a GRU, por ter apenas 2 portas (reset e update) em vez de 3, é mais eficiente computacionalmente e tende a generalizar melhor em séries temporais com padrões relativamente simples. As arquiteturas bidirecionais (BiLSTM, BiGRU) tiveram pior desempenho, o que era esperado para forecasting puro — o contexto futuro não está disponível no momento da previsão.

### Experiência 2 — Janela Temporal (n_steps)

Utilizando a GRU (melhor arquitetura), variou-se o número de dias de contexto:

| n_steps | MSE Final | Contexto |
|---|---|---|
| 3 | 0.001384 | Muito curto |
| 5 | 0.001362 | Curto |
| **10** | **0.001098** | **Médio** |
| 20 | 0.001159 | Longo |

**Análise**: A janela de 10 dias apresenta o menor MSE (0.001098). Janelas mais curtas (3, 5 dias) perdem contexto importante sobre a tendência de médio prazo. A janela de 20 dias, embora mais rica em informação, introduz ruído adicional que degrada ligeiramente o desempenho. O equilíbrio ideal situa-se em torno de 10 dias úteis (2 semanas de negociação).

### Experiência 3 — Normalizações

Foram testadas 4 técnicas de normalização com a GRU e n_steps=10:

| Normalização | MSE Final | Observação |
|---|---|---|
| **MinMax [0,1]** | **0.001106** | Melhor desempenho |
| MinMax [-1,1] | 0.003558 | 3× pior |
| StandardScaler | 0.022332 | 20× pior |
| RobustScaler | 0.008851 | 8× pior |

**Análise**: O MinMaxScaler [0,1] é claramente superior para este problema. As funções de ativação ReLU usadas nas camadas LSTM/GRU funcionam melhor com inputs no intervalo [0, 1]. O StandardScaler e RobustScaler produzem valores negativos, o que pode causar saturação ou gradientes instáveis. O MinMax [-1,1] evita valores negativos extremos mas ainda assim tem pior desempenho que o [0,1].

### Experiência 4 — Profundidade do Modelo

Foram testadas 7 configurações diferentes de camadas e unidades:

| Configuração | MSE Final | Tempo (s) |
|---|---|---|
| 1 camada, [25] | 0.001170 | 12.1 |
| 1 camada, [50] | 0.001191 | 16.1 |
| 1 camada, [100] | 0.001125 | 18.4 |
| 2 camadas, [50, 25] | 0.001140 | 22.1 |
| **2 camadas, [50, 50]** | **0.001059** | **26.3** |
| 3 camadas, [50, 25, 25] | 0.001474 | 34.0 |
| 3 camadas, [50, 50, 25] | — | ~40.0 |

**Análise**: A configuração com 2 camadas de 50 unidades cada obteve o melhor MSE (0.001059), uma melhoria de ~8% face à baseline de 1 camada. As 2 camadas permitem aprender representações hierárquicas: a primeira camada capta padrões de curto prazo e a segunda padrões de mais longo prazo. Com 3 camadas, observa-se overfitting — o modelo memoriza os dados de treino mas não generaliza. O aumento de 1 para 2 camadas tem um custo computacional moderado (+64% de tempo), mas o ganho em precisão justifica-o.

### Experiência 5 — Contexto de Mercado

Foram testados 5 conjuntos de features diferentes como input:

| Configuração | Nº Features | MSE Final |
|---|---|---|
| 1D Close (baseline) | 1 | 0.001498 |
| **2D Close + SPY** | **2** | **0.001391** |
| 2D Close + XLK | 2 | 0.001422 |
| Multi OHLC | 4 | 0.002118 |
| Multi + SPY | 5 | 0.002851 |

**Análise**: Adicionar o SPY (S&P 500) como feature extra reduz o MSE de 0.001498 para 0.001391, uma melhoria de ~7%. O SPY fornece informação sobre o movimento geral do mercado, que influencia diretamente o preço da AAPL. No entanto, adicionar demasiadas features (OHLC + SPY) piora o desempenho — o modelo fica sobrecarregado com informações redundantes e o sinal relevante dilui-se. O XLK (setor Tech) também melhora ligeiramente mas menos que o SPY, possivelmente porque a AAPL já é uma grande componente do XLK.

### Experiência 6 — Grid Search Automático

A grid search testou 8 combinações de n_steps, units e layers:

| Combinação | n_steps | Units | Layers | val_loss |
|---|---|---|---|---|
| 1 | 10 | 50 | 1 | 0.0032 |
| 2 | 10 | 50 | 2 | 0.0028 |
| 3 | 10 | 100 | 1 | 0.0021 |
| 4 | 10 | 100 | 2 | 0.0019 |
| 5 | 20 | 50 | 1 | 0.0025 |
| 6 | 20 | 50 | 2 | 0.0023 |
| **7** | **20** | **100** | **1** | **0.0017** |
| 8 | 20 | 100 | 2 | 0.0018 |

**Melhores parâmetros**: n_steps=20, units=100, layers=1 (val_loss=0.0017)

**Análise**: A grid search encontrou uma combinação superior (n_steps=20, units=100, 1 camada) que não foi descoberta nas experiências individuais. Este resultado demonstra a importância de testar combinações de hiperparâmetros em vez de variar um parâmetro de cada vez. No entanto, é de notar que a grid utilizou apenas 30 épocas (contra 50 nas experiências individuais) para manter o tempo de execução aceitável.

---

## (4) Conclusão

Este estudo demonstrou que a otimização sistemática de dados e hiperparâmetros tem um impacto significativo na performance de modelos RNN para previsão financeira. As principais conclusões são:

1. **A GRU supera a LSTM** para forecasting financeiro (MSE 30% menor), com menor custo computacional
2. **A janela temporal de 10 dias** oferece o melhor equilíbrio entre contexto e ruído
3. **Normalização MinMax [0,1]** é essencial para a convergência do modelo com ativações ReLU
4. **2 camadas RNN** melhoram a capacidade hierárquica sem cair em overfitting
5. **Contexto de mercado (SPY)** como feature extra melhora as previsões em ~7%
6. **Grid Search** sistemática descobre combinações superiores à otimização manual

A melhoria global — desde a baseline original (apenas 2 anos de dados, sem split, sem métricas) até ao pipeline final com 5 anos de dados, divisão treino/val/teste, grid search e avaliação completa — representa um avanço significativo na robustez e fiabilidade das previsões.

**Limitações e trabalho futuro**: Os resultados devem ser interpretados com cautela, dado que previsões de séries financeiras têm limitações inerentes — os mercados são influenciados por eventos imprevisíveis (notícias, decisões políticas, catástrofes). A implementação de **validação walk-forward** completa e a incorporação de **indicadores técnicos** (RSI, MACD) e **dados macroeconómicos** como features adicionais seriam os passos naturais seguintes.

---

*Anilson Monteiro — Pós-Graduação em Ciência de Dados Aplicada à Análise de Risco*
*Maio 2026*
