# Page Rank

O PageRank é um algoritmo que mede a importância relativa de cada nó dentro de um grafo. A importância de um nó é determinada pela quantidade e qualidade dos links que apontam para ele.

Os resultados apresentados na imagem e gerados código a seguir mostram os valores finais de PageRank para os nós A, B, C e D.


``` python exec="off"

graph = {
    "C": ["D", "A"],
        "A": ["C"],
        "D": ["C", "B"],
        "B": ["A"]
}

```

---

## Codigo do Page Rank

=== "code"

    ``` python exec="off" 
    --8<-- "./docs/PageRank/pagerank.py"
    ```

---

## Interpretação dos Resultados

- **Resumo numérico:** Os valores de PageRank resultantes são normalizados (somam aproximadamente 1). Por exemplo, em uma execução típica obtém-se algo como: `C ≈ 0.378`, `A ≈ 0.302`, `D ≈ 0.198`, `B ≈ 0.122`.
- **Por que C aparece em primeiro:** C recebe links de múltiplas fontes importantes (por exemplo A e D) e participa de ciclos que preservam e redistribuem PageRank, o que amplifica seu peso.
- **A versus D e B:** A fica em segundo por receber contribuições de nós com peso; D tem importância intermediária porque recebe de C mas também distribui parte do seu peso; B é o menor porque tem menos fontes de entrada e recebe menos fluxo de importância.
- **Efeito do damping factor (d):** Com `d = 0.85` usamos 85% do fluxo vindo de repasses pelos links e 15% de teletransporte uniforme. Diminuir `d` torna os valores mais próximos entre os nós (mais aleatoriedade); aumentar `d` enfatiza mais a estrutura de links.
- **Dangling nodes:** Nós sem saídas têm seu PageRank redistribuído igualmente entre todos os nós. No grafo de exemplo todos os nós têm saídas, portanto esse efeito não foi decisivo aqui.
- **Uso prático:** Ordenar nós por PageRank indica quais nós são mais centrais/influentes no grafo; útil para priorização, resumo ou análise de influência. Para decisões operacionais, foque nos nós com maior PageRank (C e A no exemplo).
- **Validação e sensibilidade:** Verifique se os valores somam ~1 (sanidade) e experimente variar `d` ou modificar arestas para testar robustez do ranking.

Se desejar, posso adicionar automaticamente os resultados de uma execução (valores numéricos concretos) ao documento, ou gerar um gráfico com os valores de PageRank.

