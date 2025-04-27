O seguinte projeto foi feito durante o Bootcamp Avanti e em conjunto com [Augusto](https://github.com/AugustoCSF) e [Giulia](https://github.com/GiuBuonafina).

# 🍊 Análise e Predição da Qualidade de Laranjas

Este projeto tem como objetivo analisar um conjunto de dados de laranjas e desenvolver modelos de machine learning para prever a qualidade dos frutos.

## Tecnologias e Ferramentas

- Python.
- Pandas.
- Numpy.
- Matplotlib.
- Seaborn.
- Scikit-learn.
- Google Colab.



## 📋Etapas do projeto

- Descrição dos dados.
- Análise exploratória de dados (EDA).
    - Análise univariada.
    - Análise bivariada.
    - Análise multivariada.

- Análise comparativa de modelos.
    - Pré-processamento de dados.
    - Transformação dos dados.
    - Modelagem.
    - Resultados.

## 📄 Descrição dos dados
O dataset foi retirado do [Kaggle](https://www.kaggle.com/datasets/shruthiiiee/orange-quality) e possui 241 entradas 11 variáveis mapeadas em um dicionário representado pela seguinte tabela:

| Variável                  | Descrição                          | Tipo         | Subtipo   |
|----------------------------|------------------------------------|--------------|-----------|
| Tamanho (cm)               | Tamanho da laranja, em centímetros. | Quantitativa | Contínua  |
| Peso (g)                   | Peso da laranja, em gramas.         | Quantitativa | Contínua  |
| Brix (Doçura)              | Nível de doçura da laranja.         | Quantitativa | Contínua  |
| pH (Acidez)                | Nível de acidez.                   | Quantitativa | Contínua  |
| Maciez (1-5)               | Nível de maciez da laranja.         | Qualitativa  | Ordinal   |
| Tempo de Colheita (dias)   | Tempo necessário para colheita.    | Quantitativa | Discreta  |
| Maturação (1-5)            | Estágio de amadurecimento da laranja. | Qualitativa | Ordinal   |
| Cor                        | Tom de cor da laranja.             | Qualitativa  | Nominal   |
| Variedade                  | Variedade da laranja.              | Qualitativa  | Nominal   |
| Manchas (S/N)              | Quantidade de danos na laranja.    | Qualitativa  | Nominal   |
| Qualidade (1-5)            | Nível de qualidade da laranja.     | Qualitativa  | Ordinal   |

## 🔍 Análise exploratória de dados (EDA).

Esta etapa foi dividida em quatro níveis de análise: resumo estatístico, univariada, bivariada e multivariada.

---
### Resumo estatístico.
Uso da biblioteca pandas para checar contagem de variáveis e valor mais frequente das variáveis qualitativas.

Nas variáveis quantitativas, foram analisadas a contagem, média, desvio padrão, valor mínimo e máximo, e porcentagens entrequartis.

**Principais observações:**
- O peso médio das laranjas é de 205.12 gramas com desvio padrão de 56.46 gramas, indicando uma variedade grande de pesos.
- A qualidade média é de 3.8 e a mediana 4.0, indicando uma maioria de laranjas de alta qualidade.

---
### Análise Univariada

Análise individual de cada variável:

- **Variáveis quantitativas contínuas** (`Tamanho`, `Peso`, `Brix`, `pH`):
  - Histogramas para observar a distribuição dos dados.
  - Boxplots para detectar outliers e assimetrias.

- **Variáveis qualitativas** (`Cor`, `Variedade`, `Manchas`, `Maturação`, `Maciez`):
  - Gráficos de barras para análise da frequência de categorias.

**Principais observações:**
- A variável `Brix` apresentou uma tendência à direita (frutas menos doces são mais comuns).
- Laranjas com qualidade a partir do valor 4, laranjas mais maduras e laranjas sem manchas são mais populares.
- Cores mais intensas são mais comuns.
- As laranjas costumam ter `pH` baixo e `Tempo de Colheita` precoce.

---

### Análise Bivariada

Estudo da relação entre duas variáveis, focando a variável-alvo `Qualidade`:

- **Quantitativas vs. Qualidade**:
  - **Boxplots** e **violinplots** para visualizar como as variáveis numéricas variam em função da qualidade.

- **Qualitativas vs. Qualidade**:
  - **Gráficos de barras**, **heatmaps**, **distribuição relativa e conjunta** e **tabelas de contingência** para entender a distribuição das categorias em relação aos níveis de qualidade.

**Principais insights:**
- Laranjas mais doces (`Brix` mais alto) tendem a ter melhor qualidade.
- `Maturação` e `Maciez` estão positivamente associadas à qualidade.
 - Frutas maiores e mais pesadas tendem a ser menos doces e mais ácidas, resultando em menor qualidade.
- O `Tempo de Colheita (dias)` influencia bastante o `Tamanho`, `Peso`, `Brix (Douçura)` e `pH (Acidez)` da fruta.
- Frutas **sem manchas** ou com **mínimos danos** têm maior chance de atingir a qualidade máxima (5.0).

---

### Análise Multivariada

Análise envolvendo múltiplas variáveis simultaneamente:

- Matriz de dispersão entre variáveis quantitativas.
- Heatmaps de variáveis qualitativas por qualidade média.

**Observações:**
- Laranjas de alta qualidade tendem a ser menores, mais leves e mais doces.
- Frutas colhidas mais cedo tendem a ser mais doces e de melhor qualidade.

---

### Perfil Ideal da Fruta de Alta Qualidade

Com base nas análises realizadas, foi possível identificar as principais características associadas às frutas de melhor qualidade:

| Característica            | Perfil Ideal               |
|----------------------------|-----------------------------|
| **Cor**                    | Deep Orange                |
| **Tempo de Colheita (dias)**| Precoce  |
| **Brix (Doçura)**           | Maior que 10               |
| **pH (Acidez)**             | Entre 3.0 e 3.5             |
| **Maciez (1-5)**            | Baixa                      |
| **Manchas (S/N)**           | Não apresenta manchas (N)   |

---


## 🛠️ Análise Comparativa de Modelos

### Pré-processamento de Dados

- Tratamento de dados faltantes, discrepantes e outliers.

---

### Transformação dos Dados
- Separação em variáveis preditoras (`X`) e alvo (`y`).
- Normalização de variáveis quantitativas com `MinMax` e codificação de variáveis qualitativas com `OneHotEncoder` e `LabelEncoder`.


---

### Modelagem

- Validação cruzada com validação de Monte Carlo (`Shuffle Split`) com 30 repetições.
- Tuning de hiperparâmetros.

Avaliamos duas abordagens possíveis para o problema: classificação e regressão. A razão disso está no fato da variável `Qualidade` poder ter interpretada tanto de forma ordinal, como de forma contínua, pois a mesma não possui unidade de medida, os valores não indicam haver proporcionalidade e não há indicação de uniformidade nas distâncias entre valores.

Modelos de ``classificação`` treinados:

- Regressão Logística
- Árvores de Decisão
- Random Forest
- XGBoost
- k-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

Modelos de ``regressão`` treinados:

- Regressão Linear
- Árvores de Decisão
- Random Forest
- XGBoost
- k-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

Cada modelo foi ajustado utilizando validação cruzada para garantir a robustez das comparações.

---

### 🏆 Resultados

#### Classificação
| Métrica              | Regressão Logística | Random Forest | SVM     |
|----------------------|---------------------|---------------|---------|
| **fit_time (mean)**   | 3.463963            | 17.674526     | **0.590797** |
| **score_time (mean)** | 0.023806            | 0.037426      | **0.022650** |
| **Acurácia (mean)**   | 0.675510            | **0.713605**      | 0.672109 |
| **Precisão (mean)**   | 0.495765            | **0.569153**      | 0.493271 |
| **Revocação (mean)**  | 0.459605            | 0.463372      | **0.488256** |
| **F1-Score (mean)**   | 0.456323            | **0.473864**      | 0.466037 |

- O modelo com melhor desempenho geral foi o Random Forest. Apesar de apresentar o maior tempo de treinamento (18.17s), seu ganho em desempenho compensa o custo computacional.

#### Regressão

| Métrica              | Regressão Linear | Random Forest | SVM  |
|----------------------|-----------------------|---------------------|-----------|
| **fit_time (mean)**   | **0.423952**      | 11.402286           | 0.483032  |
| **score_time (mean)** | 0.090062      | 0.037744            | **0.019023**  |
| **MAE (mean)**        | 0.537487      | **0.452455**            | 0.496318  |
| **MSE (mean)**        | 0.529881      | **0.444028**            | 0.488996  |
| **R2 (mean)**         | 0.451662       | **0.549476**            | 0.498653  |
| **MAPE (mean)**       | 0.192084       | **0.178521**            | 0.186592  |

- O modelo que apresentou melhor desempenho geral foi o Random Forest Regressor. Apesar de seu alto custo computacional, com o maior tempo de treinamento médio, seus resultados indicam uma maior capacidade preditiva em relação aos demais modelos.

---

## 💬 Conclusão

Em conclusão, os resultados obtidos indicam que a previsão da qualidade das laranjas pode ser abordada de forma eficaz tanto como um problema de regressão quanto de classificação e que o modelo Random Forest é uma escolha eficaz independentemente da abordagem adotada.

---
