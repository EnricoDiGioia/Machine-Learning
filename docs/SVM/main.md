``` python exec="on" html="1"
--8<-- "./docs/SVM/SVM.py"
```

**Interpretação dos resultados**

- Os quatro painéis mostram como diferentes kernels do SVM separam as classes.
- Kernel `linear`: uma fronteira simples (reta). Se as classes estiverem bem separadas por uma linha, este é o mais apropriado.
- Kernel `sigmoid`: produz fronteiras curvas simples; tende a ser menos estável como escolha padrão.
- Kernel `poly`: gera fronteiras polinomiais mais complexas; pode capturar curvaturas, mas também pode sobreajustar se o grau for alto.
- Kernel `rbf`: produz fronteiras não-lineares flexíveis e frequentemente captura padrões locais melhor que os demais.
- Se as regiões das classes estiverem muito misturadas em todos os painéis, isso indica que, visualmente, as classes não são bem separáveis com os parâmetros e visualização atuais.
- Fronteiras muito recortadas ou muito complexas podem sinalizar sobreajuste; fronteiras muito suaves podem indicar underfitting.

**Conclusão** `rbf` tem a melhor separação visual; `linear` é aceitável; `poly` aparenta sobreajuste.