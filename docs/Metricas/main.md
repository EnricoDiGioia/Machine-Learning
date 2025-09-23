## Métricas de Avaliação - Árvore de Decisão

### Métricas Utilizadas

**Accuracy:** Percentual de acertos totais do modelo
**Precision (Weighted):** Proporção de predições positivas corretas, ponderada por classe
**F1-Score (Weighted):** Média harmônica entre Precision e Recall, balanceada para multiclasse
**Matriz de Confusão:** Tabela que mostra predições corretas e incorretas por classe

### Por que essas métricas?

- **Accuracy**: Baseline fundamental para comparação
- **Precision**: Evita falsos positivos (ex: classificar Petrol como Hybrid)
- **F1-Score**: Balanceia performance em dataset com classes desbalanceadas (Hybrid é menos comum)
- **Matriz de Confusão**: Permite análise detalhada de erros por classe específica

**Weighted Average**: Pondera cada classe pelo número de amostras, adequado para classificação multiclasse (Diesel, Petrol, Hybrid).

### Árvore de Decisão com Métricas

``` python exec="on" html="1"
--8<-- "./docs/Metricas/arvore-metricas.py"
```

### Matriz de Confusão Detalhada

``` python exec="on" html="1"
--8<-- "./docs/Metricas/matriz-confusao.py"
```

