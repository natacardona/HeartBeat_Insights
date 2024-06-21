### Requerimientos:

- Se debe desarrollar un modelo de clasificación con el conjunto de datos entregado en el ejercicio anterior, el cual debe predecir la presencia de la enfermedad cardíaca en pacientes. Se debe entregar un archivo .py con el desarrollo del modelo. (Test Split 10%)

- Evalúa el rendimiento del modelo con un Split test de 20%, 13% y 5%: Genera un reporte de clasificación incluyendo la precisión, recall y score para cada una.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Cargar los datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columnas = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 
            'oldpeak', 'slope', 'ca', 'thal', 'target']

data = pd.read_csv(url, names=columnas)
data.replace('?', pd.NA, inplace=True)
data.dropna(inplace=True)

# Convertir a tipo de datos adecuado
data = data.astype(float)

# Separar características y etiqueta
X = data.drop('target', axis=1)
y = data['target'].apply(lambda x: 1 if x > 0 else 0)  # Convertir a binario

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Función para entrenar y evaluar el modelo
def entrenar_y_evaluar(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    print(f"Evaluación con test_size={test_size}:")
    print(classification_report(y_test, y_pred))

# Entrenamiento y evaluación inicial con test split del 10%
print("Entrenamiento inicial y evaluación con test split del 10%:")
entrenar_y_evaluar(X_scaled, y, test_size=0.1)

# Evaluar el modelo con diferentes tamaños de conjuntos de prueba
for test_size in [0.2, 0.13, 0.05]:
    entrenar_y_evaluar(X_scaled, y, test_size)
```

### Normalización de la tabla dimensión jerarquizada GEOGRAFIA

Desnormalizamos en varias tablas siguiendo la tercera forma normal

Tabla País:

# País

| IdPaís | NombrePaís |
|--------|------------|
|        |            |

# Departamento

| IdDepartamento | NombreDepartamento | IdPaís (llave foránea) |
|----------------|--------------------|------------------------|
|                |                    |                        |

# Ciudad

| IdCiudad | NombreCiudad | IdDepartamento (llave foránea) |
|----------|--------------|--------------------------------|
|          |              |                                |

# Barrio

| IdBarrio | NombreBarrio | IdCiudad (llave foránea) |
|----------|--------------|--------------------------|
|          |              |                          |

### Respuestas a las preguntas de negocio:

a. ¿Cuántos usuarios "Pacientes" fueron admitidos en nuestras clínicas los últimos 3 años?
Para esta pregunta, asumimos que tenemos una tabla Admisiones con al menos las siguientes columnas:

IdPaciente
FechaAdmisión
IdClinica

El script SQL para contar los pacientes admitidos en los últimos 3 años sería:

```
SELECT COUNT(DISTINCT IdPaciente) AS TotalPacientes
FROM Admisiones
WHERE FechaAdmisión >= DATEADD(YEAR, -3, GETDATE());
```

b. ¿Cuántos pacientes fueron al servicio de urgencias adultos, en los últimos 4 trimestres?
Asumimos que la tabla Admisiones también tiene la columna TipoServicio y FechaAdmisión para capturar el tipo de servicio y la fecha de admisión, respectivamente.

El script SQL para contar los pacientes que fueron al servicio de urgencias adultos en los últimos 4 trimestres sería:

```
SELECT COUNT(DISTINCT IdPaciente) AS TotalPacientesUrgenciasAdultos
FROM Admisiones
WHERE TipoServicio = 'Urgencias Adultos'
AND FechaAdmisión >= DATEADD(QUARTER, -4, GETDATE());
```