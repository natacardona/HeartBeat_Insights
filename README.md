### Setup técnico:

```
pip install pandas scikit-learn
```

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

| id_pais | nombre_pais |
|---------|-------------|
|         |             |

# Departamento

| id_departamento | nombre_departamento | id_pais (llave foránea) |
|-----------------|---------------------|-------------------------|
|                 |                     |                         |

# Ciudad

| id_ciudad | nombre_ciudad | id_departamento (llave foránea) |
|-----------|---------------|---------------------------------|
|           |               |                                 |

# Barrio

| id_barrio | nombre_barrio | id_ciudad (llave foránea) |
|-----------|---------------|---------------------------|
|           |               |                           |

### Respuestas a las preguntas de negocio:

a. ¿Cuántos usuarios "Pacientes" fueron admitidos en nuestras clínicas los últimos 3 años?
Para esta pregunta, creamos una tabla Admisiones con al menos las siguientes columnas:

id_paciente
fecha_admision
id_clinica
tipo_servicio

El script SQL para contar los pacientes admitidos en los últimos 3 años sería:

```
SELECT COUNT(DISTINCT id_paciente) AS total_pacientes
FROM admisiones
WHERE fecha_admision >= DATEADD(YEAR, -3, GETDATE());
```

b. ¿Cuántos pacientes fueron al servicio de urgencias adultos, en los últimos 4 trimestres?

El script SQL para contar los pacientes que fueron al servicio de urgencias adultos en los últimos 4 trimestres sería:

```
SELECT COUNT(DISTINCT id_paciente) AS total_pacientes_urgencias_adultos
FROM admisiones
WHERE tipo_servicio = 'Urgencias Adultos'
AND fecha_admision >= DATEADD(QUARTER, -4, GETDATE());
```

### Diseño de un modelo dimensional lógico basado en el modelo conceptual


# Tabla de Hechos: Admisiones

| id_admision | id_paciente | id_clinica | id_servicio | fecha_admision | tipo_servicio | id_tiempo |
|-------------|-------------|------------|-------------|----------------|---------------|-----------|
|             |             |            |             |                |               |           |

# Tabla Dimensión: Pacientes

| id_paciente | nombre_paciente | fecha_nacimiento | genero |
|-------------|-----------------|------------------|--------|
|             |                 |                  |        |

# Tabla Dimensión: Clínica

| id_clinica | nombre_clinica | direccion | id_geografia |
|------------|----------------|-----------|--------------|
|            |                |           |              |

# Tabla Dimensión: Servicios

| id_servicio | tipo_servicio                       |
|-------------|-------------------------------------|
|             |                                     |

# Tabla Dimensión: Geografía

| id_geografia | id_barrio |
|--------------|-----------|
|              |           |

# Tabla Dimensión: Tiempo

| id_tiempo | fecha | año | trimestre | mes | dia |
|-----------|-------|-----|-----------|-----|-----|
|           |       |     |           |     |     |

## Scripts SQL para las preguntas (a y b)

a. Total de pacientes en los últimos 3 años:

```
SELECT COUNT(DISTINCT f.id_paciente) AS total_pacientes
FROM admisiones f
JOIN tiempo t ON f.id_tiempo = t.id_tiempo
WHERE t.fecha >= DATEADD(YEAR, -3, GETDATE());
```

b. Pacientes en el servicio de urgencias adultos en los últimos 4 trimestres:

```
SELECT COUNT(DISTINCT f.id_paciente) AS total_pacientes_urgencias_adultos
FROM admisiones f
JOIN tiempo t ON f.id_tiempo = t.id_tiempo
JOIN servicios s ON f.id_servicio = s.id_servicio
WHERE s.tipo_servicio = 'Urgencias Adultos'
AND t.fecha >= DATEADD(QUARTER, -4, GETDATE());
```

### Generar una tabla con los datos de usuario (paciente)

```
CREATE TABLE paciente (
    id_paciente INT PRIMARY KEY,
    tipo_doc VARCHAR(10),
    identificacion VARCHAR(20),
    primer_nombre VARCHAR(50),
    primer_apellido VARCHAR(50),
    edad INT,
    fecha_registro DATE
);
```

-- Datos de ejemplo:

```
INSERT INTO paciente (id_paciente, tipo_doc, identificacion, primer_nombre, primer_apellido, edad, fecha_registro)
VALUES
(1, 'cc', '123456', 'Sandra', 'Moreno', 25, '2024-04-30'),
(2, '01', '123456', 'Sandra', 'Moreno', 25, '2024-04-29'),
(3, 'cc', '654321', 'Juan', 'Perez', 30, '2024-05-01'),
(4, '01', '654321', 'Juan', 'Perez', 30, '2024-05-01'),
(5, 'cc', '789012', 'Maria', 'Lopez', 28, '2024-06-01');

```

### Reflejar los datos de pacientes de dos fuentes oficiales en una dimensión

-- Crear tabla de pacientes unificada
```
CREATE TABLE paciente_unificado (
    id_paciente INT PRIMARY KEY,
    tipo_doc VARCHAR(10),
    identificacion VARCHAR(20),
    primer_nombre VARCHAR(50),
    primer_apellido VARCHAR(50),
    edad INT,
    fecha_registro DATE
);
```

-- Insertar registros únicos de pacientes
```
INSERT INTO paciente_unificado (id_paciente, tipo_doc, identificacion, primer_nombre, primer_apellido, edad, fecha_registro)
SELECT DISTINCT 
    ROW_NUMBER() OVER (ORDER BY identificacion) AS id_paciente,
    tipo_doc,
    identificacion,
    primer_nombre,
    primer_apellido,
    edad,
    MIN(fecha_registro) AS fecha_registro
FROM (
    SELECT tipo_doc, identificacion, primer_nombre, primer_apellido, edad, fecha_registro
    FROM consulta_externa

    UNION

    SELECT tipo_doc, identificacion, primer_nombre, primer_apellido, edad, fecha_registro
    FROM urgencias
) AS unioned_pacientes
GROUP BY tipo_doc, identificacion, primer_nombre, primer_apellido, edad;
```

### Justificación:

- Llaves: La llave primaria id_paciente se genera como un número de fila único para asegurar que cada paciente tenga un identificador único.
- Nivel de granularidad: Cada registro en paciente_unificado representa un paciente único.
- Claridad: Los campos son claros y concisos, y se evita la duplicación de registros.
- Vigencias y eliminaciones lógicas: Se puede implementar un proceso de una ETL para mantener los registros actualizados y manejar los cambios en la información de los pacientes.