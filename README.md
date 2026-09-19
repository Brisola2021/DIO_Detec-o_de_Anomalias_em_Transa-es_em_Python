# 💳 Detecção de Anomalias em Transações com Python

## 📌 Sobre o projeto

Este projeto tem como objetivo explorar técnicas de **Machine Learning aplicadas à detecção de anomalias em transações financeiras**, com foco na identificação de possíveis operações fraudulentas.

Em cenários financeiros, um dos principais desafios é trabalhar com **bases de dados desbalanceadas**, nas quais a quantidade de transações legítimas é muito maior do que a quantidade de transações classificadas como fraude.

Nesse contexto, o projeto utiliza **técnicas de preparação dos dados, balanceamento de classes, modelos de classificação e métricas específicas** para avaliar a capacidade dos algoritmos em identificar transações anômalas.

O dataset utilizado é o `creditcard.csv`, carregado diretamente a partir de uma fonte disponibilizada pelo TensorFlow. 

---

## 🎯 Objetivos

O projeto foi desenvolvido com os seguintes objetivos:

* 🔎 Analisar dados de transações financeiras;
* 🚨 Identificar possíveis transações fraudulentas;
* ⚖️ Trabalhar com o problema de **desbalanceamento de classes**;
* 🧹 Realizar transformações e preparação dos dados;
* 🤖 Aplicar diferentes algoritmos de Machine Learning;
* 📊 Avaliar o desempenho dos modelos;
* 📈 Utilizar métricas como **Accuracy, Precision, Recall, F1-Score e AUC**;
* 📉 Analisar curvas **ROC** e **Precision-Recall**;
* 🌳 Avaliar a importância das variáveis utilizadas pelos modelos.

---

## 🧠 Contexto: por que detectar anomalias?

Em uma base de transações financeiras, a maioria das operações tende a ser legítima, enquanto uma parcela muito menor pode representar fraude.

Isso cria um problema de **desbalanceamento de classes**.

Imagine, por exemplo, uma base contendo milhares de transações legítimas e poucas transações fraudulentas. Um modelo poderia apresentar uma alta taxa de acerto simplesmente classificando quase todas as operações como legítimas.

Por isso, em problemas de fraude, **não basta analisar somente a acurácia**.

É importante observar métricas como:

* 🎯 **Precision:** entre as transações classificadas como fraude, quantas realmente eram fraude;
* 🔍 **Recall:** entre todas as fraudes existentes, quantas foram identificadas;
* ⚖️ **F1-Score:** equilíbrio entre Precision e Recall;
* 📈 **ROC-AUC:** capacidade do modelo de separar as classes em diferentes limiares.

O projeto utiliza essas métricas para realizar uma avaliação mais completa dos modelos. 

---

# 🛠️ Tecnologias e bibliotecas

O projeto utiliza Python e diversas bibliotecas voltadas para análise de dados e Machine Learning:

| Tecnologia              | Utilização                             |
| ----------------------- | -------------------------------------- |
| 🐍 **Python**           | Linguagem principal                    |
| 🐼 **Pandas**           | Manipulação e análise dos dados        |
| 🔢 **NumPy**            | Operações matemáticas                  |
| 📊 **Matplotlib**       | Criação das visualizações              |
| 🤖 **Scikit-learn**     | Modelagem e avaliação                  |
| ⚖️ **Imbalanced-learn** | Técnicas de balanceamento              |
| 🌲 **Random Forest**    | Classificação                          |
| 🚀 **XGBoost**          | Classificação e análise de importância |
| 🔬 **SHAP**             | Interpretação das previsões            |

Essas bibliotecas aparecem na estrutura de importação do projeto e são utilizadas ao longo das etapas de preparação, modelagem, avaliação e interpretação. 

---

# 📂 Dataset

O projeto utiliza o dataset:

**`creditcard.csv`**

O arquivo é carregado diretamente através da seguinte fonte:

```python
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)
```

Após o carregamento, é possível visualizar as primeiras linhas e analisar a distribuição da variável `Class`. 

A variável `Class` é utilizada como variável-alvo da classificação:

```python
x = df.drop("Class", axis=1)
y = df["Class"]
```

Ou seja:

* `X` → variáveis utilizadas para realizar as previsões;
* `y` → classe que representa o resultado que o modelo deve aprender a identificar. 

---

# 🧹 Preparação e transformação dos dados

Uma das etapas do projeto consiste na transformação da variável relacionada ao valor da transação.

Foi criada uma nova variável utilizando transformação logarítmica:

```python
df["Amount_log"] = np.log1p(df["Amount"])
```

Além disso, foi aplicado o **StandardScaler** para realizar a padronização de `Amount`:

```python
scaler = StandardScaler()

df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
```

Essa etapa faz parte da preparação dos dados antes da aplicação dos modelos de Machine Learning. 

---

# ✂️ Separação entre treinamento e teste

Os dados são divididos em conjuntos de treinamento e teste utilizando `train_test_split`:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)
```

A configuração utilizada reserva **30% dos dados para teste**, enquanto os demais dados são utilizados no treinamento do modelo. 

---

# 🤖 Modelo de Regressão Logística

Como primeiro modelo de classificação, o projeto utiliza **Logistic Regression**:

```python
model = LogisticRegression(max_iter=1000)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
```

Após o treinamento, o modelo realiza previsões sobre o conjunto de teste e seus resultados são avaliados através do relatório de classificação. 

---

# 📈 Avaliação através da curva ROC

O projeto também calcula as probabilidades previstas pelo modelo:

```python
y_probs = model.predict_proba(x_test)[:, 1]
```

A partir dessas probabilidades é construída a **curva ROC**, permitindo analisar a relação entre:

* **False Positive Rate**
* **True Positive Rate**

```python
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
```

Além da visualização da curva, é calculado o **AUC Score**:

```python
roc_auc_score(y_test, y_probs)
```



---

# 🎯 Precision-Recall

Para problemas envolvendo classes desbalanceadas, a relação entre **Precision e Recall** também é importante.

O projeto utiliza:

```python
precision, recall, _ = precision_recall_curve(
    y_test,
    y_probs
)
```

E apresenta graficamente a relação entre as duas métricas. 

Essa análise permite observar o comportamento do modelo quando diferentes níveis de classificação são considerados.

---

# ⚖️ Técnicas de Balanceamento

Um dos principais focos do projeto é trabalhar com o **desbalanceamento das classes**.

Foram exploradas duas abordagens:

### 🔽 Under-sampling

No Under-sampling, parte da classe majoritária é reduzida para equilibrar sua quantidade em relação à classe minoritária.

No projeto:

```python
fraudes = df[df["Class"] == 1]

nao_fraudes = df[
    df["Class"] == 0
].sample(
    n=len(fraudes),
    random_state=42
)

df_under = pd.concat([
    fraudes,
    nao_fraudes
])
```

A quantidade de transações não fraudulentas é reduzida para corresponder à quantidade de transações classificadas como fraude. 

---

### 🔼 Over-sampling com SMOTE

Também é utilizado o **SMOTE — Synthetic Minority Over-sampling Technique**.

```python
smote = SMOTE(random_state=42)

x_resampled, y_resampled = smote.fit_resample(
    x,
    y
)
```

A ideia é aumentar a representação da classe minoritária por meio da geração de exemplos sintéticos. 

---

# 🌳 Random Forest

O projeto também utiliza o algoritmo **Random Forest Classifier**:

```python
rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
```

Um ponto importante é a utilização de:

```python
class_weight="balanced"
```

que permite ao modelo considerar o desbalanceamento entre as classes durante o treinamento. 

Depois do treinamento, o modelo realiza previsões:

```python
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)
```

e seus resultados são avaliados através do `classification_report`. 

---

# 🔗 Pipeline de Machine Learning

O projeto também apresenta o conceito de **Pipeline**, combinando a padronização dos dados com o modelo de classificação:

```python
Pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])
```

Depois, o pipeline é treinado utilizando os dados de treinamento:

```python
pipeline.fit(x_train, y_train)

y_pred_pipeline = pipeline.predict(x_test)
```



Essa abordagem ajuda a organizar as etapas de transformação e modelagem dentro de um único fluxo.

---

# 🎚️ Ajuste do Threshold

Outro conceito importante abordado no projeto é o ajuste do **limiar de decisão (threshold)**.

Foi definido:

```python
thresholds = 0.3

y_pred_custom = (
    y_probs >= thresholds
).astype(int)
```

Em vez de utilizar somente o limite padrão de classificação, o projeto demonstra como um threshold diferente pode ser aplicado às probabilidades produzidas pelo modelo. 

Essa abordagem é particularmente relevante em problemas de detecção de fraude, nos quais pode ser necessário analisar o equilíbrio entre identificar operações suspeitas e gerar classificações incorretas.

---

# 🚀 XGBoost

Outra abordagem utilizada é o **XGBoost Classifier**:

```python
xgb = XGBClassifier(
    scale_pos_weight=10,
    use_label_encoder=False,
    eval_metric="logloss"
)
```

O parâmetro:

```python
scale_pos_weight=10
```

é utilizado no contexto do tratamento da classe positiva durante o treinamento. 

---

# 🔍 Feature Importance

O projeto também explora a importância das variáveis utilizadas pelo XGBoost:

```python
importance = xgb.feature_importances_
```

Esses valores são apresentados graficamente através do Matplotlib:

```python
plt.bar(range(len(importance)), importance)

plt.title("Feature Importance")

plt.show()
```



Essa análise permite observar quais características tiveram maior participação nas decisões do modelo.

---

# ⚙️ Grid Search e otimização de hiperparâmetros

O projeto também apresenta uma etapa de busca de hiperparâmetros utilizando `GridSearchCV`.

São definidos diferentes valores para parâmetros como:

```python
param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100]
}
```

A avaliação utiliza **Recall** como métrica:

```python
GridSearchCV(
    XGBClassifier(eval_metric="logloss"),
    param_grid,
    scoring="recall",
    cv=3
)
```



Essa etapa demonstra como testar diferentes configurações do modelo para investigar combinações de hiperparâmetros e seu impacto na capacidade de identificação da classe de interesse.

---

# 🔬 Interpretabilidade com SHAP

Por fim, o projeto apresenta o uso de **SHAP** para interpretação das previsões do modelo:

```python
explainer = shap.Explainer(xgb)

shap_values = explainer(
    x_test[:100]
)

shap.plots.bar(shap_values)
```



A utilização do SHAP acrescenta uma camada de **interpretabilidade**, permitindo analisar a contribuição das variáveis para as previsões realizadas pelo modelo.

---

# 🧩 Fluxo do projeto

De forma resumida, o projeto segue este fluxo:

```text
📥 Carregamento dos dados
        ↓
🔎 Análise da distribuição das classes
        ↓
🧹 Tratamento e transformação dos dados
        ↓
⚖️ Estratégias de balanceamento
        ↓
✂️ Separação em treino e teste
        ↓
🤖 Treinamento dos modelos
        ↓
📊 Avaliação das previsões
        ↓
📈 ROC e Precision-Recall
        ↓
🎚️ Ajuste de Threshold
        ↓
🌳 Random Forest
        ↓
🚀 XGBoost
        ↓
⚙️ Otimização de hiperparâmetros
        ↓
🔍 Feature Importance
        ↓
🔬 Interpretabilidade com SHAP
```

---

# 📚 Principais conceitos estudados

Durante o desenvolvimento deste projeto, foram trabalhados conceitos importantes de **Data Science e Machine Learning**, como:

* 🐍 Python para análise de dados;
* 🐼 Pandas;
* 🔢 NumPy;
* 🧹 Data preprocessing;
* 📊 Análise exploratória;
* ⚖️ Desbalanceamento de classes;
* 🔽 Under-sampling;
* 🔼 Over-sampling;
* 🧬 SMOTE;
* 📏 StandardScaler;
* 🤖 Regressão Logística;
* 🌳 Random Forest;
* 🚀 XGBoost;
* 📈 ROC-AUC;
* 🎯 Precision;
* 🔍 Recall;
* ⚖️ F1-Score;
* 🎚️ Threshold;
* ⚙️ Grid Search;
* 🔬 SHAP;
* 📊 Feature Importance;
* 🔗 Pipelines de Machine Learning.

---

# 💼 Aplicação no contexto de dados

A detecção de anomalias em transações é um exemplo de aplicação de **Machine Learning em problemas financeiros**, nos quais a análise automatizada pode auxiliar na identificação de padrões diferentes daqueles observados nas transações consideradas normais.

Neste projeto, o foco está principalmente no processo de **preparação dos dados, tratamento do desbalanceamento, construção dos modelos e avaliação dos resultados**, permitindo compreender como diferentes estratégias podem ser utilizadas em um problema de classificação com classes desbalanceadas.

---

# 🚀 Conclusão

Este projeto representa uma aplicação prática de técnicas de **Machine Learning para análise de transações financeiras**, explorando desde o carregamento e tratamento dos dados até técnicas de balanceamento, classificação, avaliação e interpretabilidade.

O principal aprendizado está em compreender que problemas de detecção de fraude não devem ser analisados apenas pela quantidade de previsões corretas. O **desbalanceamento das classes**, a escolha das métricas, o ajuste do threshold, a avaliação dos modelos e a interpretação das variáveis são elementos importantes para construir uma análise mais consistente.

Dessa forma, o projeto reúne diferentes conceitos de **Data Science, Machine Learning e análise de dados**, formando um estudo prático sobre como modelos preditivos podem ser aplicados à identificação de transações anômalas.

---

## 👨‍💻 Projeto desenvolvido para estudos em Data Science

**Tema:** Detecção de Anomalias em Transações
**Linguagem:** Python
**Área:** Data Science / Machine Learning
**Foco:** Classificação, desbalanceamento de dados e detecção de fraudes

---

