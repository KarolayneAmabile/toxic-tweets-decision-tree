# Classificação de tweets tóxicos com Árvore de Decisão e Regressão Logística

Projeto de **classificação** que prevê se um tweet em português é **tóxico** e, se for, **de que tipo**: obscenidade,
insulto, homofobia ou ódio identitário (racismo, misoginia e xenofobia agrupados). São 5 classes ao todo, contando
"nenhuma". A partir do texto (TF-IDF), compara uma **Árvore de Decisão implementada do zero** com a
**Regressão Logística treinada por Gradiente Descendente**.

## O que está sendo feito

Tudo está no notebook [Projeto2_ToLD-BR_Classificacao.ipynb](Projeto2_ToLD-BR_Classificacao.ipynb):

1. **Estudo introdutório:** distribuição das classes (base muito desbalanceada: ~80% não tóxicos), tamanho dos textos,
   exemplos e remoção de duplicados.
2. **Preparação:** rótulo único (categoria mais votada com ao menos 2 de 3 anotadores), divisão estratificada
   60/20/20 (treino/validação/teste) e TF-IDF com 5.000 termos, ajustado só no treino.
3. **Modelos:**
   - Regressão Logística com Gradiente Descendente (`SGDClassifier`): sem regularização, Ridge, Lasso e Elastic Net;
   - Árvore de Decisão própria (CART com impureza de Gini), com treino (`fit`) e predição (`predict` e `predict_proba`).
4. **Análise de consistência:** os dois algoritmos treinados com 30%, 60% e 90% do treino.
5. **Curvas de aprendizado:** quatro para a Regressão Logística e uma para a árvore, variando a profundidade até a
   altura limite.
6. **Avaliação no teste:** precisão, recall, F1 micro e F1 macro; matrizes de confusão dos três modelos mais
   significativos; e a visão binária (tóxico × não tóxico).

**Principal resultado:** a Regressão Logística com Elastic Net foi o melhor modelo, com F1 macro de 0,493 nas 5 classes
e de 0,707 em tóxico × não tóxico.

## Como rodar

**No Google Colab:** abra o notebook e execute todas as células. A base é baixada direto do GitHub do ToLD-Br e não
precisa de nenhum arquivo local.

**Localmente:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib jupyter
jupyter notebook Projeto2_ToLD-BR_Classificacao.ipynb
```

A execução completa leva cerca de 5 minutos.

## Dataset

**ToLD-Br** (*Toxic Language Dataset for Brazilian Portuguese*): 21.000 tweets em português anotados por 3 anotadores
em 6 categorias de toxicidade.

- Repositório: <https://github.com/joaoaleite/ToLD-Br>
- Artigo: LEITE, J. A.; SILVA, D. F.; BONTCHEVA, K.; SCARTON, C. *Toxic Language Detection in Social Media for
  Brazilian Portuguese: New Dataset and Multilingual Analysis*. AACL-IJCNLP, 2020.

Créditos da base aos autores do ToLD-Br.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `Projeto2_ToLD-BR_Classificacao.ipynb` | Notebook principal do projeto |
| `IA_Classificação.ipynb` | Notebook inicial (estudo introdutório, split e TF-IDF), que serviu de base para o principal |
| `enunciado.txt` | Enunciado do projeto |

## Créditos

Trabalho desenvolvido para a disciplina de **Inteligência Artificial** do curso de Bacharelado em Ciência da Computação
do Instituto Federal de Goiás (IFG), ministrada pelo
[Prof. Dr. Daniel Xavier de Sousa](https://www.escavador.com/sobre/3446148/daniel-xavier-de-sousa).

Feito pelas alunas **Karolayne Amábile Brito Borges** e **[Ana Laura Machado Pereira](https://github.com/aanalaura)**.

## Licença

O código deste projeto está sob a [licença MIT](LICENSE). A base de dados segue a licença e os termos de uso da sua fonte original.
