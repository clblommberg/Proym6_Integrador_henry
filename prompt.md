1. **Variables Ordinales:**
    - **DIABETES:** Parece ser una variable binaria (SÍ/NO). Puedes aplicar Label Encoding ya que tiene un orden implícito (SÍ > NO).
    - **HOSPITALIZACIÓN ULTIMO MES:** Parece ser una variable numérica, pero dado que representa el número de hospitalizaciones en el último mes, podría tratarse como ordinal. Podrías aplicar Label Encoding si hay un orden lógico.
    - **CUP (Número de copas):** Si este representa un número discreto de copas, podrías tratarlo como ordinal y aplicar Label Encoding.

2. **Variables Nominales:**
    - **ANTIBIOTICO UTILIZADO EN LA PROFILAXIS:** Dado que hay diferentes antibióticos, esta variable parece ser nominal. Deberías aplicar One-Hot Encoding.
    - **TIPO DE CULTIVO:** Si hay diferentes tipos de cultivos, deberías aplicar One-Hot Encoding.
    - **AGENTE AISLADO:** Dado que hay diferentes agentes aislados, esta variable parece ser nominal. Deberías aplicar One-Hot Encoding.
    - **PATRÓN DE RESISTENCIA:** Dado que hay diferentes patrones de resistencia, esta variable parece ser nominal. Deberías aplicar One-Hot Encoding.

3. **Variables numéricas:**
    - **EDAD, PSA, VOLUMEN PROSTÁTICO, NUMERO DE MUESTRAS TOMADAS, NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACIÓN INFECCIOSA, DIAS HOSPITALIZACION MQ, DIAS HOSPITALIZACIÓN UPC:** Estas parecen ser variables numéricas y puedes tratarlas utilizando la normalización estándar.

En el marco teórico proporcionado, las variables dependientes son las que se están tratando de predecir o explicar, mientras que las variables independientes son aquellas que se utilizan para predecir o explicar la variable dependiente. En este caso, la variable dependiente es la "Indicación de biopsias prostáticas" y las variables independientes son diversas características del paciente y del procedimiento.

Aquí hay una aclaración sobre las variables dependientes e independientes:

- **Variable Dependiente:** Indicación de biopsias prostáticas (BIOPSIA).

- **Variables Independientes:**
  1. **Antecedentes del paciente:**
     - EDAD (Edad del paciente).
     - DIABETES (Indicador de diabetes).
     - HOSPITALIZACIÓN ULTIMO MES (Indicador de hospitalización en el último mes).

  2. **Morbilidad asociada al paciente:**
     - CUP (Uso de catéter urinario al momento de la biopsia).

  3. **Antecedentes relacionados con la toma de la biopsia:**
     - ENF. CRONICA PULMONAR OBSTRUCTIVA (Indicador de enfermedad crónica pulmonar obstructiva).
     - VOLUMEN PROSTATICO (Indicador de volumen prostático mayor a 40 cm3).
     - PSA (Concentración de PSA en la sangre).
     - BIOPSIAS PREVIAS (Indicador de biopsias previas).
     - ANTIBIOTICO UTILIZADO EN LA PROFILAXIS (Tipo de antibiótico utilizado en la profilaxis).
     - NUMERO DE MUESTRAS TOMADAS (Número de muestras tomadas en la biopsia).

  4. **Complicaciones Infecciosas:**
     - BIOPSIA (Resultado de la biopsia).
     - NUMERO DE DIAS POST BIOPSIA EN QUE SE PRESENTA LA COMPLICACIÓN INFECCIOSA (Días post-biopsia en los que se presenta la complicación infecciosa).
     - FIEBRE (Indicador de presencia de fiebre).
     - ITU (Indicador de infección de tracto urinario).
     - TIPO DE CULTIVO (Tipo de cultivo encontrado).
     - AGENTE AISLADO (Tipo de agente aislado).
     - PATRON DE RESISTENCIA (Indicador de patrón de resistencia).

  5. **HOSPITALIZACIÓN:**
     - HOSPITALIZACION (Indicador de hospitalización).
     - DIAS HOSPITALIZACION MQ (Días de hospitalización médico quirúrgico).
     - DIAS HOSPITALIZACIÓN UPC (Días de hospitalización en estado crítico).

### Marco Teórico:

En este estudio, se pretende evaluar la relación entre estas variables y la indicación de biopsias prostáticas. Se utilizarán técnicas estadísticas y econometría para analizar los datos y derivar conclusiones sobre la asociación de estas variables con la necesidad de biopsias.

### Hipótesis Nula (H0) y Hipótesis Alternativa (H1):

Se han establecido hipótesis nulas y alternativas para cada una de las variables independientes, donde la hipótesis nula (H0) sugiere que no hay efecto significativo, mientras que la hipótesis alternativa (H1) sugiere que hay un efecto significativo.

Este enfoque proporciona una base sólida para aplicar técnicas estadísticas y econometría en la evaluación de la relación entre las variables y la indicación de biopsias prostáticas, permitiendo así una comprensión más profunda de los factores que podrían influir en esta indicación.


Para mejorar el rendimiento del modelo de árbol de decisión, puedes considerar las siguientes sugerencias:

1. **Ajuste de Hiperparámetros:**
   - Experimenta con valores más altos para `max_depth` en tu espacio de búsqueda de hiperparámetros. Esto permitirá que el árbol sea más profundo y pueda capturar patrones más complejos en los datos.

   ```python
   espacio_de_parametros = {
       'max_depth': [10, 15, 20],
       'min_samples_leaf': [32, 64, 128],
       'criterion': ['gini', 'entropy']
   }
   ```

2. **Explorar Otros Hiperparámetros:**
   - Además de `max_depth` y `min_samples_leaf`, también puedes explorar otros hiperparámetros como `min_samples_split` y `max_features`. Estos hiperparámetros pueden influir en la estructura del árbol y su capacidad para generalizar.

   ```python
   espacio_de_parametros = {
      'max_depth': [8, 10, 12, 15],
      'min_samples_leaf': [10, 12, 14, 16], #16, 32, 64, 128, 256
      'min_samples_split': [2, 5, 8, 10],
      'max_features': [None, 'sqrt', 'log2', 0.5, 0.7],
      'criterion': ['gini', 'entropy']
   }
   ```

3. **Regularización:**
   - Añadir regularización puede ser beneficioso. Puedes experimentar con valores más altos para el parámetro `ccp_alpha`, que controla la complejidad del árbol y ayuda a prevenir el sobreajuste.

   ```python
   espacio_de_parametros = {
       'max_depth': [10, 15, 20],
       'min_samples_leaf': [32, 64, 128],
       'criterion': ['gini', 'entropy'],
       'ccp_alpha': [0.0, 0.1, 0.2]
   }
   ```

4. **Balanceo de Clases:**
   - Si las clases están desbalanceadas, considera técnicas de balanceo de clases, como el parámetro `class_weight` en el modelo de árbol de decisión.

   ```python
   modelo = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, class_weight='balanced', random_state=SEED)
   ```

5. **Prueba con Otros Modelos:**
   - Dependiendo de la complejidad y la naturaleza de tus datos, también podrías probar otros modelos de clasificación, como Random Forests o Gradient Boosted Trees, para comparar su rendimiento.

6. **Ingeniería de Características:**
   - Explora la posibilidad de realizar ingeniería de características para crear nuevas variables que puedan mejorar la capacidad predictiva del modelo.

Implementa estos cambios y realiza pruebas para evaluar cómo afectan al rendimiento del modelo. Recuerda siempre validar tu modelo en un conjunto de prueba independiente para asegurarte de que las mejoras observadas no sean específicas del conjunto de entrenamiento.


```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split

# Seed para reproducibilidad
SEED = 42

# Generar datos de ejemplo (puedes reemplazar esto con tus datos reales)
X, y = make_classification(n_samples=1000, n_features=36, n_informative=33, n_redundant=0, n_classes=9, random_state=SEED)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Normalización de datos con nombres de características
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=[f'feature_{i}' for i in range(X_train.shape[1])])

# Convertir X_train_scaled de nuevo a DataFrame (asegúrate de tener las columnas correctamente etiquetadas)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=[f'feature_{i}' for i in range(X_train_scaled.shape[1])])

# Análisis de correlación y selección de características
corr_matrix = X_train_scaled_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_train_selected = X_train_scaled_df.drop(columns=to_drop, axis=1)

# Reducción de dimensionalidad con PCA
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)

# Pesos de clase personalizados
weights_arr = (
    (0.01, 0.01, 0.98),
    (0.01, 0.05, 0.94),
    (0.2, 0.1, 0.7),
    (0.33, 0.33, 0.33),
)

# Ajuste de hiperparámetros usando GridSearchCV (ajusta según tus necesidades)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Crear un conjunto de subgráficos
fig, axs = plt.subplots(len(weights_arr), 2, figsize=(12, 6 * len(weights_arr)))

for i, weights in enumerate(weights_arr):
    clf = LogisticRegression(max_iter=1000, class_weight={0: weights[0], 1: weights[1], 2: weights[2]})
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train_pca[:, :2], y_train.ravel())

    best_clf = LogisticRegression(max_iter=1000, class_weight={0: weights[0], 1: weights[1], 2: weights[2]})
    best_clf.fit(X_train_pca[:, :2], y_train.ravel())

    # Convertir X_test a DataFrame
    X_test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])

    # Obtener las columnas seleccionadas en el conjunto de prueba
    X_test_selected_columns = X_test_df.columns.intersection(X_train_selected.columns)

    # Selecciona las mismas características en el conjunto de prueba que se eliminaron en el conjunto de entrenamiento
    X_test_selected = X_test_df[X_test_selected_columns]

    # Normaliza el conjunto de prueba utilizando el mismo escalador que se utilizó en el conjunto de entrenamiento
    X_test_scaled = scaler.transform(X_test_selected)

    # Aplica la reducción de dimensionalidad con PCA en el conjunto de prueba
    X_test_pca = pca.transform(X_test_scaled)[:, :2]

    # Visualizar la decisión del clasificador en el conjunto de prueba
    plot_decision_regions(X_test_pca, y_test, best_clf, ax=axs[i, 0])
    axs[i, 0].set_title(f"Decision function with weights {weights}")

    # Visualizar la decisión del clasificador en el conjunto de entrenamiento
    plot_decision_regions(X_train_pca[:, :2], y_train, best_clf, ax=axs[i, 1])
    axs[i, 1].set_title(f"Decision function with weights {weights} (Training)")

# Ajustar los subgráficos
plt.tight_layout()

# Mostrar los gráficos
plt.show()

```



```python
# Convertir las etiquetas de clase a listas
train_class_labels = y_train.biopsia.unique()
test_class_labels = y_test.biopsia.unique()
print(train_class_labels)
print(test_class_labels)
# Crear una figura y ejes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# Graficar la distribución de clases en conjunto de entrenamiento
for class_label in train_class_labels:
    class_data = X_train.loc[X_train.index == class_label]
    axes[0].bar(class_label, class_data.values.flatten(), color=['blue', 'red'])

axes[0].set_title('Training Set Class Distribution')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Proportion')

# Graficar la distribución de clases en conjunto de prueba
for class_label in test_class_labels:
    class_data = X_test.loc[X_test.index == class_label]
    axes[1].bar(class_label, class_data.values.flatten(), color=['blue', 'red'])

axes[1].set_title('Test Set Class Distribution')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Proportion')

plt.tight_layout()
plt.show()

```