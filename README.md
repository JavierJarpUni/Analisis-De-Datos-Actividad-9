# Sistema de Recomendaciones SVD

**Análisis de Datos - Actividad 9**  
Universidad - Sistema de Recomendación E-commerce usando Singular Value Decomposition

## Descripción del Proyecto

Este proyecto implementa un sistema de recomendaciones basado en **filtrado colaborativo** usando el algoritmo **SVD (Singular Value Decomposition)** para analizar patrones de compra en datos de e-commerce y generar recomendaciones personalizadas para usuarios.

### Objetivos Académicos

- Demostrar la implementación práctica de algoritmos de machine learning
- Aplicar técnicas de filtrado colaborativo en sistemas de recomendación
- Desarrollar una interfaz web interactiva para visualizar resultados
- Analizar y explicar el comportamiento de algoritmos de recomendación

## Tecnologías Utilizadas

| Tecnología          | Versión  | Propósito                    |
| ------------------- | -------- | ---------------------------- |
| **Python**          | 3.11+    | Lenguaje principal           |
| **Streamlit**       | Latest   | Framework web interactivo    |
| **Pandas**          | Latest   | Manipulación de datos        |
| **NumPy**           | 1.26.4   | Computación numérica         |
| **Scikit-surprise** | Latest   | Algoritmos de recomendación  |
| **Plotly**          | Latest   | Visualizaciones interactivas |
| **Pickle**          | Built-in | Serialización de modelos     |

## Dataset

El proyecto utiliza un dataset de comportamiento de compras e-commerce que incluye:

- **Clientes**: 3,900+ usuarios únicos
- **Productos**: 1,000+ artículos diferentes
- **Transacciones**: 3,900+ compras registradas
- **Características**: Edad, género, categorías, precios, ratings, historial

### Métricas del Dataset

- Total de clientes únicos
- Total de productos únicos
- Número de transacciones
- Ingresos totales generados

## Algoritmo SVD

### ¿Qué es SVD?

**Singular Value Decomposition** es una técnica de factorización matricial que:

1. **Encuentra patrones ocultos** en los datos de usuario-producto
2. **Factoriza la matriz** en componentes más simples
3. **Identifica usuarios similares** basándose en preferencias
4. **Predice puntuaciones** para productos no comprados

### Ventajas del SVD

- Maneja datos dispersos efectivamente
- Captura relaciones latentes complejas
- Escalable para grandes datasets
- Reduce dimensionalidad preservando información

## Instalación y Configuración

### Prerrequisitos

```bash
- Python 3.11 o superior
- pip (gestor de paquetes de Python)
- Git (opcional, para clonar el repositorio)
```

### Pasos de Instalación

1. **Clonar el repositorio** (si aplica):

```bash
git clone https://github.com/JavierJarpUni/Analisis-De-Datos-Actividad-9.git
cd streamlit-recommendation-app
```

2. **Crear entorno virtual**:

```bash
python -m venv .venv
source .venv/bin/activate  # En macOS/Linux
# o
.venv\Scripts\activate     # En Windows
```

3. **Instalar dependencias**:

```bash
pip install -r requirements.txt
```

4. **Verificar estructura de archivos**:

```
streamlit-recommendation-app/
├── streamlit_app.py          # Aplicación principal
├── data/
│   ├── shopping_behavior_updated.csv
│   └── models/
│       ├── svd_model.pkl
│       └── item_similarity_matrix.csv
├── requirements.txt
└── README.md
```

## �️ Ejecución de la Aplicación

### Método 1: Ejecución Directa

```bash
streamlit run streamlit_app.py --server.port 8503
```

### Método 2: Con Entorno Virtual Específico

```bash
python -m streamlit run streamlit_app.py --server.port 8503
```

### Acceso a la Aplicación

Una vez ejecutada, la aplicación estará disponible en:

- **URL Local**: http://localhost:8503
- **URL de Red**: http://[tu-ip]:8503

## 🎮 Guía de Uso

### 1. **Selección de Cliente**

- Usa el selector desplegable para elegir un Customer ID
- Revisa el perfil completo del cliente seleccionado

### 2. **Configuración de Recomendaciones**

- Ajusta el número de recomendaciones (3-15)
- Haz clic en "🚀 Generar Recomendaciones SVD"

### 3. **Análisis de Resultados**

- Revisa las recomendaciones generadas
- Analiza las puntuaciones y niveles de confianza
- Explora la visualización interactiva

### 4. **Interpretación**

- **Puntuación Alta (>4.0)**: Confianza alta de recomendación
- **Puntuación Media (3.0-4.0)**: Confianza moderada
- **Puntuación Baja (<3.0)**: Confianza limitada

## Características de la Interfaz

### Página Principal

- **Perfil del Cliente**: Información demográfica y de compras
- **Explicación del Algoritmo**: Descripción educativa de SVD
- **Métricas del Dataset**: Estadísticas generales

### Sección de Recomendaciones

- **Lista Detallada**: Productos recomendados con información completa
- **Sistema de Puntuación**: Scores SVD con interpretación
- **Visualización**: Gráfico interactivo de puntuaciones

### Información por Producto

- Nombre y categoría del producto
- Precio promedio y rating
- Popularidad (número de compras)
- Nivel de confianza de la recomendación

## Metodología

### Preprocesamiento de Datos

1. **Limpieza de datos**: Manejo de valores faltantes
2. **Segmentación**: Creación de grupos de edad
3. **Score de interacción**: Combinación de rating, precio y historial

### Entrenamiento del Modelo

1. **Matriz usuario-producto**: Creación de matriz dispersa
2. **Factorización SVD**: Entrenamiento con parámetros optimizados
3. **Validación**: Evaluación de performance del modelo

### Generación de Recomendaciones

1. **Filtrado**: Exclusión de productos ya comprados
2. **Predicción**: Cálculo de scores para productos nuevos
3. **Ranking**: Ordenamiento por puntuación predicha

## Fundamentos Teóricos

### Filtrado Colaborativo

El filtrado colaborativo se basa en la premisa de que usuarios con preferencias similares en el pasado tendrán gustos parecidos en el futuro.

### SVD en Sistemas de Recomendación

SVD descompone la matriz usuario-producto R en tres matrices:

```
R = U × Σ × V^T
```

Donde:

- **U**: Factores latentes de usuarios
- **Σ**: Valores singulares (importancia de factores)
- **V^T**: Factores latentes de productos

## Casos de Uso

### Para Estudiantes

- Entender algoritmos de machine learning aplicados
- Analizar sistemas de recomendación reales
- Explorar técnicas de visualización de datos

### Para Profesores

- Demostrar conceptos teóricos con ejemplos prácticos
- Evaluar comprensión de algoritmos de recomendación
- Mostrar aplicaciones industriales de ML

### Para Desarrolladores

- Implementación práctica de sistemas de recomendación
- Integración de modelos ML en aplicaciones web
- Técnicas de visualización interactiva

## Limitaciones y Mejoras Futuras

### Limitaciones Actuales

- Dataset limitado a un dominio específico
- Modelo entrenado con parámetros fijos
- Sin actualización en tiempo real

### Posibles Mejoras

- [ ] Implementar más algoritmos (Matrix Factorization, Deep Learning)
- [ ] Añadir filtrado híbrido (colaborativo + contenido)
- [ ] Implementar evaluación A/B testing
- [ ] Agregar retroalimentación de usuarios
- [ ] Optimización de hiperparámetros automática

---

### Explanation of the Code:

- **Imports**: The necessary libraries are imported, including Streamlit, pandas, and pickle.
- **Load the Model**: The SVD model is loaded from the `svd_model.pkl` file.
- **Streamlit Interface**: The app has a title and a sidebar where users can input their customer ID.
- **Get Recommendations**: When the user clicks the button, the app fetches recommendations for the specified customer ID using the `get_svd_recommendations` function.
- **Display Recommendations**: The recommendations are displayed in the main area of the app.

### Running the Application:

To run the Streamlit application, navigate to the directory where `app.py` is located and execute the following command in your terminal:

```bash
streamlit run app.py
```

This will start a local server, and you can view the application in your web browser at `http://localhost:8501`.
