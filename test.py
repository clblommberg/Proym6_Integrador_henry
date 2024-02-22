

# 1. Suponga que la relación entre dos variables (x,y) es modelada usando una función lineal con 
# los resultados que aparece a continuación. ¿Cuál de las siguientes interpretaciones es correcta?
"""
y = Bo + B12

Bo= 0.1, B1-2.3, R2 = 0.2.

1-(x,y) siguen una tendencia lineal con alta dispersión y pendiente negativa.
2-(x,y) siguen una tendencia lineal con baja dispersión y pendiente negativa.
3-(x,y) siguen una tendencia lineal con alta dispersión y pendiente positiva.
4- (x,y) siguen una tendencia lineal con baja dispersión y pendiente positiva.                 RPTA
"""

# 2. Suponiendo que tenemos un conjunto de datos con poco ruido, ¿cuándo un modelo presenta 
# subajuste (underfitting)?
"""
1- El error de entrenamiento es bajo, pero el error de prueba es alto.
2- El error de entrenamiento es alto, pero el error de prueba es bajo.                 RPTA
3- Tanto los errores de entrenamiento como los de prueba son bajos. 
4- Tanto el error del entrenamiento como el de la prueba son altos.
"""


# 3. ¿Cuál de los siguientes modelos de clasificación de la siguiente imagen corresponde a un ejemplo de 
# sobreajuste?

"""
1 -A
2 -B
3 -C
4 -D                RPTA
"""

# 4. ¿Cuál es la afirmación correcta respecto a datos nulos y ceros en un conjunto de datos de entrenamiento?
"""
1- Un valor nulo (NaN) representa un dato faltante, mientras que un valor cero puede representar una 
respuesta numérica.                RPTA
2- Un valor nulo (NaN) se reemplaza siempre por un cero en el proceso de limpieza de datos.
3- Un valor nulo (NaN) es equivalente a un valor cero, ambos representan datos faltantes.
4- Un valor nulo (NaN) es interpretado por pandas como un valor cero.
"""
# 5. ¿Cuál de las siguientes afirmaciones relativa a algoritmos de aprendizaje supervisado es VERDADERA?

"""
1- Los algoritmos de clasificación sólo pueden generar predicciones en base a variables de entrada numéricas.
2- Los algoritmos de regresión sólo permiten predecir variables numéricas continuas.                RPTA
3- Los algoritmos de clasificación sólo permiten predecir categorías binarias.
4- Los algoritmos de regresión requieren datos de entrenamiento no etiquetados.
"""


#6. El índice de calor en un lugar particular (1) depende de la temperatura T *1/1

# (en grados Celsius), la humedad H y la velocidad del viento V (en m/s). Un modelo de regresión múltiple 
# para el índice de calor (1) se describe mediante la ecuación que se muestra en la imagen. 
# ¿Cuál de las siguientes interpretaciones del modelo es correcta?

# I(H,T,V) = 6H + T − 0,4V

"""
1- Por cada unidad en que se incrementa la velocidad del viento (V), el índice de✓ calor 
disminuye en promedio 0.4 unidades.                RPTA
2- Por cada unidad en que se incrementan la temperatura (T) y la humedad (H) el índice de calor 
aumenta en promedio 7 unidades.
3- Por cada unidad en que se incrementa la velocidad del viento (V), la humedad (H) aumenta en 
promedio 0.4 unidades.
4- Por cada unidad en que se incrementa la velocidad del viento (V), el índice de calor aumenta en 
promedio 0.4 unidades.
"""

# 7. ¿Cuál de las siguientes estrategias es utilizada para evitar el sobreajuste en un modelo de clasificación
# de árbol de decisión?
"""
1- Forzar la creación de hojas puras.
2- Limitar la profundidad máxima del árbol.                RPTA
3- Aumentar la cantidad de nodos.
4- Aumentar la complejidad del árbol.
"""


# 8. ¿Qué es el uso de técnicas de remuestreo en el balanceo de clases? *

"""
1- Una técnica que consiste en eliminar ejemplos de la clase mayoritaria.
2- Una técnica que consiste en generar datos sintéticos para la clase minoritaria mediante interpolación 
entre los puntos de datos cercanos.
3- Una técnica que consiste en asignar un peso diferente a los errores de clasificación de las diferentes 
clases.
4- Una técnica que consiste en modificar la distribución de los datos mediante ✓ técnicas como oversampling, 
under-sampling, y generación de datos sintéticos.                RPTA
"""

# 9. Como característica de un buen proceso de clustering se puede definir:

"""
1- Los clústeres o grupos formados deben tener el mismo número de registros para garantizar un buen 
balanceo de datos.

2- La distancia entre cada centroide y los registros de cada uno de ellos debe ser igual a cero.

3- Los registros del clúster deben ser lo más cohesionados posible, es decir, que el centroide represente 
muy bien el clúster.               RPTA

4- El entrenamiento se hace con un porcentaje de los datos que puede ser entre el 70-80% y se debe reservar 
una cantidad de registros para evaluación.
"""


# 10. Considerando las tres gráficas de learning rate, ¿cuál de ellas parece *1/1 tener un 
# learning rate bajo y por qué?


"""
La gráfica de la izquierda porque tiene una convergencia lenta y gradual hacia el mínimo global.               RPTA
La gráfica de la derecha porque converge rápidamente al mínimo global sin oscilaciones.
La gráfica del medio porque tiene una convergencia rápida y estable hacia el mínimo global sin oscilaciones.
No se podría concluir con la información disponible.
"""


# Utilizando el notebook y las bases 10 de de datos disponibles en el siguiente link
# https://drive.google.com/drive/folders/13AQ_rKWXoJ8YORna1m0FinTvieAnJyOJ?

#10 puntos
# 11. Nuestro target es la columna 'Sales'. Si se generara una regresión lineal 
# múltiple y=f(x1,x2,x3), utilizando el 100% del dataset y trabajando con los hiperparámetros 
# por defecto, el valor de R2 se espera que esté en el intervalo:

"""
(0.93,1.00]
(0.87,0.93]               RPTA
(0.76,0.87]
(0.00,0.76]
"""


# 12. Al generar un modelo de regresión lineal simple para cada variable por separado, es decir, y1=f(x1), y2=f(x2) y y3= f(x3), con y1, y2, y3 la variable 'Sales', utilizando el 100% del dataset y trabajando con los hiperparámetros por defecto, la mayor bondad de ajuste se logra para:
"""
y1=f(x1)               RPTA
y2=f(x2)
"""

# Para los tres modelos es la misma
# 13. Siguiendo únicamente el paso a paso

#  descrito en el notebook y utilizando como variable dependiente "charges" y el resto de las 
# variables presentes en el dataframe (exceptuando "region") como variables independientes, 
# ¿Entre qué intervalo de valores se encuentra el intercepto para dicho modelo?
"""
(-13000,-11000]               RPTA
(-11000,-5000]
(-5000,0]
(0,5000)
"""

# 14. Si usted tuviese que elegir una variable para generar un modelo lineal *1/1 
# simple que explique la variable SalePrice ¿Cuál de las siguientes

#  variables utilizaría?
"""
GrLivArea               RPTA
BedroomAbvGr
OverallCond
KitchenAbvGr'
"""

# 15. Revise atentamente el bloque de código entregado. Después de efectuado el procedimiento, se pide utilizar validación cruzada, con cv=10 y aplicar árboles de decisión ¿En qué rango se encuentra el accuracy del modelo?
"""
1. [0.65, 0.7)
2. [0.7,0.8)
3. [0.8, 0.9)
4. [0.9, 0.95]
5. Menor a 0.65 o mayor a 0.95               RPTA
"""


# 16. ¿Cuál sería el valor de la exhaustividad del modelo?  RESPUESTA 0.857


"""
0.625
0.833
0.857            RPTA
0.714
"""

# 17. Considerando 1 como clase positiva, el valor de los Falsos Negativos es (escribir la respuesta 
# en número). Ejemplo de respuesta: 5               RPTA 2




#! 18. Si en el conjunto entregado, instancia y entrena un modelo KMeans con tres clústeres, el valor 
# de la inercia, estaría en el rango de: 

"""
(10000, 20000]
(0, 10000]          RPTA
(20000, 30000]
Mayor a 30000
"""


#! 19. ¿Cuáles son los mejores hiperparámetros para la SVM? *
"""
{'C': 0.01, 'coef0': 0, 'gamma': 1, 'kernel': 'sigmoid'}
{'C': 0.1, 'coef0': 0, 'gamma': 1, 'kernel': 'rbf'}
{'C': 1, 'coef0': 0, 'gamma': 1, 'kernel': 'rbf'}           RPTA
{'C': 1, 'coef0': 0, 'gamma': 1, 'kernel': 'sigmoid'}
"""

# 20. ¿La cantidad de covariables m que se utilizan en la construcción de *1/1 un árbol, 
# dentro del RandomForest, es un hiperparámetro?, de ser así

"""
¿cuál de las siguientes estrategias permitiría identificar apropiadamente el valor de m?
El número de covariables m no es un hiperparámetro, y su valor es calculado como la raíz del 
número de covariables.

El número de covariables m no es un hiperparámetro, y su valor siempre es igual a 5, 
independiente del número de covariables en el conjunto de datos.

El número de covariables m sí es un hiperparámetro, pero no puede ser obtenido mediante 
validación cruzada, si no que se debe proponer por el usuario.

El número de covariables m sí es un hiperparámetro y puede ser obtenido mediante 
validación cruzada considerando una grilla para dicho valor, seleccionando aquel que 
minimice la función de riesgo (error del modelo) escogida.             RPTA
"""
