# Streamlit Cloud Deployment Configuration

## Archivos Principales

Para desplegar en Streamlit Cloud, usa estos archivos:

### 1. Archivo Principal

- **app_cloud.py** - Versión optimizada para Streamlit Cloud con datos sintéticos
- **requirements.txt** - Dependencias mínimas necesarias

### 2. Configuración de Streamlit Cloud

En la configuración de Streamlit Cloud:

- **Main file path**: `app_cloud.py`
- **Python version**: 3.9+
- **Requirements file**: `requirements.txt`

### 3. Variables de Entorno (si necesarias)

Ninguna requerida para esta aplicación.

## Problemas Comunes y Soluciones

### Error: ModuleNotFoundError

- Asegúrate de que `requirements.txt` solo contenga las dependencias necesarias
- Evita versiones específicas a menos que sea necesario

### Error: File not found

- Usa `app_cloud.py` en lugar de `streamlit_app.py` para el despliegue
- Esta versión no depende de archivos externos

### Error: Memory limit exceeded

- La versión `app_cloud.py` usa datos sintéticos pequeños
- Evita cargar archivos CSV o modelos grandes

### Error: Build timeout

- Reduce las dependencias en `requirements.txt`
- Usa versiones estables de las librerías

## Diferencias entre Versiones

### streamlit_app.py (Local)

- Carga datos reales desde CSV
- Usa modelo SVD pre-entrenado
- Mejor para desarrollo local

### app_cloud.py (Cloud)

- Genera datos sintéticos
- Entrena modelo en tiempo real
- Optimizado para Streamlit Cloud

## Instrucciones de Despliegue

1. **Subir a GitHub**: Asegúrate de que los archivos estén en tu repositorio
2. **Conectar con Streamlit Cloud**: https://share.streamlit.io/
3. **Configurar la aplicación**:
   - Repository: tu-usuario/tu-repositorio
   - Branch: main
   - Main file path: `streamlit-recommendation-app/app_cloud.py`
4. **Deploy**: La aplicación se desplegará automáticamente

## URLs de Ejemplo

- **Repositorio**: https://github.com/JavierJarpUni/Analisis-De-Datos-Actividad-9
- **App**: https://tu-app.streamlit.app/ (después del despliegue)
