# Sistema de Recomendación E-commerce - Streamlit App

Esta aplicación web desarrollada con Streamlit implementa un sistema de recomendación completo para una plataforma de comercio electrónico, utilizando múltiples algoritmos de machine learning.

## 🚀 Características Principales

- **Recomendaciones Personalizadas**: Sistema híbrido que combina filtrado colaborativo y basado en contenido
- **Análisis de Clientes**: Perfiles detallados y segmentación RFM
- **Análisis de Productos**: Tendencias, popularidad y análisis de similitud
- **Visualizaciones Interactivas**: Dashboards con métricas clave y gráficos dinámicos
- **Interfaz Intuitiva**: Navegación fácil con múltiples páginas

## 📊 Modelos Implementados

1. **SVD (Singular Value Decomposition)**: Factorización de matrices para filtrado colaborativo
2. **Filtrado Basado en Ítems**: Recomendaciones por similitud de productos
3. **Sistema Híbrido**: Combina ambos enfoques para mayor precisión

## 🛠 Instalación y Configuración

### Prerrequisitos

- Python 3.8+
- pip

### Instalación

1. **Clona o descarga el proyecto**:

```bash
git clone <repository-url>
cd streamlit-recommendation-app
```

2. **Instala las dependencias**:

```bash
pip install -r requirements.txt
```

3. **Verifica la estructura de archivos**:

```
streamlit-recommendation-app/
├── data/
│   ├── shopping_behavior_updated.csv
│   └── models/
│       ├── svd_model.pkl
│       ├── item_similarity_matrix.csv
│       └── rfm_analysis.csv
├── src/
│   ├── app.py
│   └── components/
└── config.py
```

## 🚀 Ejecución

1. **Ejecuta la aplicación**:

```bash
cd src
streamlit run app.py
```

2. **Abre tu navegador** en: `http://localhost:8501`

## 📱 Funcionalidades de la App

### 1. Resumen General

- Métricas clave del negocio
- Distribución por categorías
- Análisis por segmentos de clientes
- Tendencias estacionales

### 2. Análisis de Clientes

- Selección de cliente específico
- Perfil detallado del cliente
- Historial de compras
- Recomendaciones personalizadas (híbrido, SVD, basado en ítems)

### 3. Análisis de Productos

- Top productos por categoría
- Análisis de productos similares
- Métricas de popularidad
- Relación precio-rating

### 4. Rendimiento del Modelo

- Estadísticas del dataset
- Análisis de sparsity
- Métricas de rendimiento

## 🔧 Configuración

Edita `config.py` para personalizar:

- Rutas de archivos
- Parámetros del modelo
- Configuración de UI

## 📈 Algoritmos Utilizados

### SVD (Singular Value Decomposition)

- **Propósito**: Filtrado colaborativo
- **Ventajas**: Maneja bien la sparsity, escalable
- **Implementación**: Scikit-Surprise

### Filtrado Basado en Ítems

- **Propósito**: Recomendaciones por similitud
- **Métrica**: Similitud coseno
- **Ventajas**: Explicable, estable

### Sistema Híbrido

- **Combinación**: 60% colaborativo + 40% contenido
- **Ventajas**: Mayor precisión y diversidad

## 🎯 Métricas de Evaluación

- **RMSE**: Error cuadrático medio
- **MAE**: Error absoluto medio
- **Cobertura**: Porcentaje de ítems recomendables
- **Diversidad**: Variedad en recomendaciones

## 🚀 Despliegue en la Nube

### Streamlit Cloud (Recomendado)

1. Sube el código a GitHub
2. Conecta con Streamlit Cloud
3. Deploy automático

### Otras opciones

- **Heroku**: Para mayor control
- **AWS/GCP**: Para escalabilidad enterprise

## 📊 Estructura de Datos

### Dataset Principal

- `Customer ID`: Identificador único del cliente
- `Item Purchased`: Producto comprado
- `Purchase Amount (USD)`: Monto de la compra
- `Review Rating`: Calificación del producto
- `Category`: Categoría del producto
- `Age`, `Gender`: Información demográfica

### Archivos del Modelo

- `svd_model.pkl`: Modelo SVD entrenado
- `item_similarity_matrix.csv`: Matriz de similitud entre productos
- `rfm_analysis.csv`: Análisis RFM de clientes

## 🔍 Troubleshooting

### Error al cargar datos

- Verifica que todos los archivos CSV y PKL estén en `/data/`
- Confirma que el modelo SVD fue entrenado correctamente

### Error de importación

- Instala todas las dependencias: `pip install -r requirements.txt`
- Verifica la versión de Python (3.8+)

### Rendimiento lento

- El primer carga puede ser lenta (caching de Streamlit)
- Para datasets grandes, considera optimizar el modelo

## 📝 Próximas Mejoras

- [ ] Integración con base de datos en tiempo real
- [ ] API REST para recomendaciones
- [ ] A/B testing para algoritmos
- [ ] Métricas de negocio avanzadas
- [ ] Sistema de feedback de usuarios

## 👥 Contribuciones

Este proyecto fue desarrollado como parte de la materia de Análisis de Datos.

## 📄 Licencia

Proyecto académico - Universidad

---

**Desarrollado para Análisis de Datos - Actividad 9**

````

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
````

This will start a local server, and you can view the application in your web browser at `http://localhost:8501`.
