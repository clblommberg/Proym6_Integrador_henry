#  Machine Learning a Servicios Médicos 
## Indicador de biopsias prostáticas.
### Preparación de Datos
- Se estandarizaron 550 registros médicos de pacientes para el análisis.
- Un 40% de los casos tenían datos faltantes en ciertos campos como mediciones de antígeno prostático.
- Se categorizaron 12 variables como factores de riesgo relevantes.
- Revisar supuestos:<br>
[1_EDA.ipynb](https://github.com/clblommberg/Proym6_Integrador_henry/blob/main/notebooks/1_EDA.ipynb)

### Modelado
- Se entrenó un modelo de Árboles de Decisión para predecir riesgo de biopsias. 
- El modelo tuvo una precisión de 41% y capacidad de detección (recall) de 52% en el conjunto de prueba.
- Sólo predijo correctamente un 27% de casos de alto riesgo en la validación.
- Revisar supuestos:<br>
[2_Modelamiento.ipynb](https://github.com/clblommberg/Proym6_Integrador_henry/blob/main/notebooks/2_Modelamiento.ipynb)<br>
[3_Entrenamiento.ipynb](https://github.com/clblommberg/Proym6_Integrador_henry/blob/main/notebooks/3_Entrenamiento.ipynb)<br>
[4_Prediciones.ipynb](https://github.com/clblommberg/Proym6_Integrador_henry/blob/main/notebooks/4_Prediciones.ipynb)

### Explicación a Usuarios Médicos
- Con la precisión actual del 41%, más de la mitad de las predicciones serían erróneas, lo cual es inaceptable.  
- Para uso clínico se debería alcanzar al menos una precisión del 80%, con recall sobre 60%.
- Estamos trabajando en conseguir datos adicionales de pacientes para mejorar la detección de los casos complejos.
- Agradecemos su paciencia; los mantendremos informados sobre el progreso en las próximas semanas.

En resumen, el desempeño actual del 41% de precisión no es suficiente para uso médico confiable. Trabajaremos en mejorar la calidad del modelo mediante la incorporación de más datos de pacientes. Por favor contáctennos ante cualquier duda!