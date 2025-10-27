import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    import config
except ImportError:
    # Fallback configuration if config.py is not found
    class Config:
        SHOPPING_DATA_PATH = "../data/shopping_behavior_updated.csv"
        SVD_MODEL_PATH = "../data/models/svd_model.pkl"
        ITEM_SIMILARITY_PATH = "../data/models/item_similarity_matrix.csv"
        RFM_ANALYSIS_PATH = "../data/models/rfm_analysis.csv"
        APP_TITLE = "Sistema de Recomendación E-commerce"
        APP_DESCRIPTION = "Análisis de comportamiento de compra y recomendaciones personalizadas"
        PAGE_ICON = "🛒"
    
    config = Config()

# Import custom components
from components.data_loader import DataLoader
from components.recommendation_engine import RecommendationEngine
from components.visualizations import Visualizations

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #2e86ab;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if 'recommendation_engine' not in st.session_state:
        st.session_state.recommendation_engine = None
    
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = None

@st.cache_data
def load_all_data():
    """Load all necessary data"""
    data_loader = DataLoader()
    
    # Load main dataset
    df = data_loader.load_shopping_data()
    if df is None:
        return None, None, None, None
    
    # Load other components
    item_similarity_df = data_loader.load_item_similarity()
    rfm_df = data_loader.load_rfm_analysis()
    svd_model = data_loader.load_svd_model()
    
    return df, item_similarity_df, rfm_df, svd_model

def show_overview_page(df, viz):
    """Show overview/dashboard page"""
    st.markdown('<h2 class="section-header">Resumen del Dataset</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clientes", f"{df['Customer ID'].nunique():,}")
    with col2:
        st.metric("Total Productos", f"{df['Item Purchased'].nunique():,}")
    with col3:
        st.metric("Total Transacciones", f"{len(df):,}")
    with col4:
        st.metric("Ingresos Totales", f"${df['Purchase Amount (USD)'].sum():,.2f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">Distribución por Categorías</h3>', unsafe_allow_html=True)
        viz.plot_category_distribution()
    
    with col2:
        st.markdown('<h3 class="section-header">Análisis por Segmento</h3>', unsafe_allow_html=True)
        viz.plot_spending_by_segment()
    
    # Additional charts
    st.markdown('<h3 class="section-header">Análisis de Precios vs Ratings</h3>', unsafe_allow_html=True)
    viz.plot_price_rating_scatter()
    
    st.markdown('<h3 class="section-header">Tendencias Estacionales</h3>', unsafe_allow_html=True)
    viz.plot_seasonal_trends()

def show_customer_analysis_page(df, rec_engine, viz):
    """Show customer analysis page"""
    st.markdown('<h2 class="section-header">Análisis de Cliente</h2>', unsafe_allow_html=True)
    
    # Customer selection
    customers = sorted(df['Customer ID'].unique())
    selected_customer = st.selectbox("Selecciona un Cliente:", customers)
    
    if selected_customer:
        # Get customer profile
        profile = rec_engine.get_customer_profile(selected_customer)
        
        if profile:
            st.markdown('<h3 class="section-header">Perfil del Cliente</h3>', unsafe_allow_html=True)
            viz.plot_customer_profile_metrics(profile)
            
            # Purchase history
            st.markdown('<h3 class="section-header">Historial de Compras Reciente</h3>', unsafe_allow_html=True)
            history_df = pd.DataFrame({
                'Producto': profile['purchase_history'],
                'Índice': range(1, len(profile['purchase_history']) + 1)
            })
            st.dataframe(history_df, use_container_width=True)
            
            # Personalized recommendations
            st.markdown('<h3 class="section-header">Recomendaciones Personalizadas</h3>', unsafe_allow_html=True)
            
            # Choose recommendation type
            rec_type = st.radio(
                "Tipo de Recomendación:",
                ["Híbrido (Recomendado)", "Solo Colaborativo (SVD)", "Solo Basado en Ítems"]
            )
            
            num_recommendations = st.slider("Número de Recomendaciones:", 5, 20, 10)
            
            if st.button("Generar Recomendaciones"):
                with st.spinner("Generando recomendaciones..."):
                    if rec_type == "Híbrido (Recomendado)":
                        recommendations = rec_engine.get_hybrid_recommendations(selected_customer, num_recommendations)
                    elif rec_type == "Solo Colaborativo (SVD)":
                        recommendations = rec_engine.get_svd_recommendations(selected_customer, num_recommendations)
                    else:
                        # For item-based, we'll use the first item from customer's history
                        if profile['purchase_history']:
                            recommendations = rec_engine.get_item_based_recommendations(
                                profile['purchase_history'][0], num_recommendations
                            )
                        else:
                            recommendations = []
                    
                    viz.plot_recommendations_table(recommendations, f"Recomendaciones {rec_type}")

def show_product_analysis_page(df, rec_engine, viz):
    """Show product analysis page"""
    st.markdown('<h2 class="section-header">Análisis de Productos</h2>', unsafe_allow_html=True)
    
    # Category analysis
    categories = ['Todos'] + sorted(df['Category'].unique())
    selected_category = st.selectbox("Selecciona una Categoría:", categories)
    
    if selected_category == 'Todos':
        st.markdown('<h3 class="section-header">Top Productos Globales</h3>', unsafe_allow_html=True)
        viz.plot_top_products()
    else:
        st.markdown(f'<h3 class="section-header">Top Productos en {selected_category}</h3>', unsafe_allow_html=True)
        viz.plot_top_products(category=selected_category)
        
        # Popular items in category
        popular_items = rec_engine.get_popular_items_by_category(selected_category, 10)
        
        if popular_items:
            st.markdown('<h3 class="section-header">Detalles de Productos Populares</h3>', unsafe_allow_html=True)
            
            items_df = pd.DataFrame(popular_items)
            items_df.columns = ['Producto', 'Compras', 'Precio Promedio', 'Rating Promedio']
            items_df['Precio Promedio'] = items_df['Precio Promedio'].apply(lambda x: f"${x:.2f}")
            items_df['Rating Promedio'] = items_df['Rating Promedio'].apply(lambda x: f"{x:.2f}/5")
            
            st.dataframe(items_df, use_container_width=True)
    
    # Item similarity analysis
    st.markdown('<h3 class="section-header">Productos Similares</h3>', unsafe_allow_html=True)
    
    products = sorted(df['Item Purchased'].unique())
    selected_product = st.selectbox("Selecciona un Producto para ver similares:", products)
    
    if selected_product and st.button("Buscar Productos Similares"):
        similar_items = rec_engine.get_item_based_recommendations(selected_product, 10)
        viz.plot_recommendations_table(similar_items, f"Productos similares a {selected_product}")

def show_model_performance_page(df):
    """Show model performance and statistics"""
    st.markdown('<h2 class="section-header">Rendimiento del Modelo</h2>', unsafe_allow_html=True)
    
    # Dataset statistics
    st.markdown('<h3 class="section-header">Estadísticas del Dataset</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Información General:**")
        st.write(f"- Total de registros: {len(df):,}")
        st.write(f"- Clientes únicos: {df['Customer ID'].nunique():,}")
        st.write(f"- Productos únicos: {df['Item Purchased'].nunique():,}")
        st.write(f"- Categorías: {df['Category'].nunique()}")
        
    with col2:
        st.markdown("**Métricas de Negocio:**")
        st.write(f"- Gasto promedio: ${df['Purchase Amount (USD)'].mean():.2f}")
        st.write(f"- Rating promedio: {df['Review Rating'].mean():.2f}/5")
        st.write(f"- Compras previas promedio: {df['Previous Purchases'].mean():.1f}")
    
    # Sparsity analysis
    user_item_matrix = df.pivot_table(
        index='Customer ID',
        columns='Item Purchased', 
        values='interaction_score',
        aggfunc='mean'
    ).fillna(0)
    
    sparsity = (1 - (user_item_matrix > 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1])) * 100
    
    st.markdown('<h3 class="section-header">Análisis de Matriz Usuario-Producto</h3>', unsafe_allow_html=True)
    st.write(f"- Dimensiones de la matriz: {user_item_matrix.shape[0]:,} usuarios × {user_item_matrix.shape[1]:,} productos")
    st.write(f"- Sparsity: {sparsity:.2f}%")
    st.write(f"- Interacciones totales: {(user_item_matrix > 0).sum().sum():,}")

def main():
    """Main application"""
    initialize_session_state()
    
    # Title
    st.markdown(f'<h1 class="main-header">{config.APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; color: #666;">{config.APP_DESCRIPTION}</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Cargando datos y modelos..."):
        df, item_similarity_df, rfm_df, svd_model = load_all_data()
    
    if df is None:
        st.error("Error al cargar los datos. Por favor, verifica que todos los archivos estén en su lugar.")
        return
    
    # Initialize components
    rec_engine = RecommendationEngine(df, svd_model, item_similarity_df)
    viz = Visualizations(df)
    
    # Sidebar navigation
    st.sidebar.title("Navegación")
    pages = {
        "Resumen General": "overview",
        "Análisis de Clientes": "customers", 
        "Análisis de Productos": "products",
        "Rendimiento del Modelo": "performance"
    }
    
    selected_page = st.sidebar.radio("Selecciona una página:", list(pages.keys()))
    
    # Show selected page
    if pages[selected_page] == "overview":
        show_overview_page(df, viz)
    elif pages[selected_page] == "customers":
        show_customer_analysis_page(df, rec_engine, viz)
    elif pages[selected_page] == "products":
        show_product_analysis_page(df, rec_engine, viz)
    elif pages[selected_page] == "performance":
        show_model_performance_page(df)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Desarrollado para Análisis de Datos**")
    st.sidebar.markdown("Universidad - Actividad 9")

if __name__ == "__main__":
    main()