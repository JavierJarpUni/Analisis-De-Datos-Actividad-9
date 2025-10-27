   # Configuration file for the recommendation system
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Data files
SHOPPING_DATA_PATH = os.path.join(DATA_DIR, "shopping_behavior_updated.csv")
SVD_MODEL_PATH = os.path.join(MODELS_DIR, "svd_model.pkl")
ITEM_SIMILARITY_PATH = os.path.join(MODELS_DIR, "item_similarity_matrix.csv")
RFM_ANALYSIS_PATH = os.path.join(MODELS_DIR, "rfm_analysis.csv")
SAMPLE_RECOMMENDATIONS_PATH = os.path.join(MODELS_DIR, "sample_recommendations.csv")

# App configuration
APP_TITLE = "Sistema de Recomendación E-commerce"
APP_DESCRIPTION = "Análisis de comportamiento de compra y recomendaciones personalizadas"
PAGE_ICON = "🛒"

# Model parameters
SVD_FACTORS = 50
N_RECOMMENDATIONS = 10
INTERACTION_WEIGHTS = {
    'rating': 0.4,
    'amount': 0.3,
    'loyalty': 0.3
}

# UI Configuration
SIDEBAR_WIDTH = 300
CHART_HEIGHT = 400
CHART_WIDTH = 600