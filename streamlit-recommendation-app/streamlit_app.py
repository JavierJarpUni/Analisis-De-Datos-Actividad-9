import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sistema de Recomendaciones SVD - Análisis de Datos",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: bold;
}
.subtitle {
    font-size: 1.2rem;
    color: #7f8c8d;
    text-align: center;
    margin-bottom: 2rem;
}
.explanation-box {
    background-color: #e8f4fd;
    border-left: 5px solid #3498db;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
}
.algorithm-box {
    background-color: #f8f9fa;
    border: 2px solid #dee2e6;
    padding: 1.5rem;
    margin: 1rem 0;
    border-radius: 10px;
}
.customer-profile {
    background-color: #fff3cd;
    border-left: 5px solid #ffc107;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
}
.recommendation-item {
    background-color: #d4edda;
    border-left: 5px solid #28a745;
    padding: 0.8rem;
    margin: 0.5rem 0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all necessary data"""
    try:
        # Load main dataset
        df = pd.read_csv("data/shopping_behavior_updated.csv")
        
        # Basic preprocessing
        df['Purchase Amount (USD)'] = pd.to_numeric(df['Purchase Amount (USD)'], errors='coerce')
        df['Review Rating'] = pd.to_numeric(df['Review Rating'], errors='coerce')
        df['Previous Purchases'] = pd.to_numeric(df['Previous Purchases'], errors='coerce')
        
        # Create customer segments
        df['Customer_Segment'] = pd.cut(df['Age'], 
                                      bins=[0, 25, 40, 60, 100], 
                                      labels=['Joven', 'Adulto', 'Maduro', 'Senior'])
        
        # Calculate interaction score
        df['interaction_score'] = (
            df['Review Rating'] * 0.4 +
            (df['Purchase Amount (USD)'] / df['Purchase Amount (USD)'].max()) * 5 * 0.3 +
            (df['Previous Purchases'] / df['Previous Purchases'].max()) * 5 * 0.3
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_svd_model():
    """Load trained SVD model"""
    try:
        with open("data/models/svd_model.pkl", 'rb') as f:
            svd_model = pickle.load(f)
        return svd_model
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo SVD: {str(e)}")
        return None

def create_user_item_matrix(df):
    """Create user-item matrix"""
    return df.pivot_table(
        index='Customer ID',
        columns='Item Purchased',
        values='interaction_score',
        aggfunc='mean'
    ).fillna(0)

def get_svd_recommendations(svd_model, df, customer_id, top_n=5):
    """Get SVD-based recommendations with explanations"""
    if svd_model is None:
        return [], "Modelo SVD no disponible"
        
    # Get all items
    all_items = df['Item Purchased'].unique()
    
    # Get items already purchased by customer
    customer_items = set(df[df['Customer ID'] == customer_id]['Item Purchased'])
    
    # Predict scores for unpurchased items
    predictions_list = []
    
    for item in all_items:
        if item not in customer_items:
            try:
                pred = svd_model.predict(customer_id, item)
                predictions_list.append((item, pred.est))
            except:
                continue
    
    # Sort and return top N
    predictions_list.sort(key=lambda x: x[1], reverse=True)
    
    explanation = f"""
    **¿Cómo funciona SVD (Singular Value Decomposition)?**
    
    1. **Análisis de patrones**: Encuentra patrones ocultos en las compras de todos los usuarios
    2. **Factorización de matriz**: Descompone la matriz usuario-producto en factores latentes
    3. **Predicción**: Predice qué tan probable es que le guste un producto no comprado
    4. **Puntuación**: Productos con mayor puntuación = mayor probabilidad de compra
    
    ✅ Cliente tiene {len(customer_items)} productos ya comprados
    ⚡ Se evaluaron {len(all_items) - len(customer_items)} productos nuevos
    🎯 Mostrando los {min(top_n, len(predictions_list))} productos con mayor probabilidad
    """
    
    return predictions_list[:top_n], explanation

def get_item_based_recommendations(item_similarity_df, df, customer_id, top_n=5):
    """Get item-based recommendations with explanations"""
    # Get customer's purchase history
    customer_items = df[df['Customer ID'] == customer_id]['Item Purchased'].unique()
    
    if len(customer_items) == 0:
        return [], "Cliente sin historial de compras"
    
    # Use the most recent or highest rated item
    latest_item = customer_items[0]  # For simplicity, use first item
    
    if latest_item not in item_similarity_df.index:
        return [], f"No se encontraron similitudes para {latest_item}"
    
    similar_items = item_similarity_df[latest_item].sort_values(ascending=False)[1:top_n+1]
    recommendations = list(zip(similar_items.index, similar_items.values))
    
    explanation = f"""
    **¿Cómo funciona Filtrado Basado en Ítems?**
    
    1. **Producto base**: Se toma como referencia "{latest_item}"
    2. **Cálculo de similitud**: Usa correlación de Pearson entre productos
    3. **Clientes similares**: Encuentra usuarios que compraron productos similares
    4. **Recomendación**: Sugiere productos que compraron usuarios con gustos similares
    
    Basado en el historial de compras: {', '.join(customer_items[:3])}
    {'...' if len(customer_items) > 3 else ''}
    """
    
    return recommendations, explanation

def get_customer_profile(df, customer_id):
    """Get customer profile"""
    customer_data = df[df['Customer ID'] == customer_id]
    
    if customer_data.empty:
        return None
    
    profile = {
        'customer_id': customer_id,
        'age': customer_data['Age'].iloc[0],
        'gender': customer_data['Gender'].iloc[0],
        'segment': customer_data['Customer_Segment'].iloc[0],
        'total_purchases': len(customer_data),
        'unique_items': customer_data['Item Purchased'].nunique(),
        'total_spent': customer_data['Purchase Amount (USD)'].sum(),
        'avg_spent': customer_data['Purchase Amount (USD)'].mean(),
        'avg_rating': customer_data['Review Rating'].mean(),
        'favorite_category': customer_data['Category'].mode().iloc[0] if len(customer_data['Category'].mode()) > 0 else 'N/A',
        'purchase_history': customer_data['Item Purchased'].unique()[:5]
    }
    
    return profile

def display_customer_profile(profile):
    """Display customer profile in a clean format"""
    st.markdown('<div class="customer-profile">', unsafe_allow_html=True)
    st.markdown(f"### 👤 Perfil del Cliente {profile['customer_id']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Edad:** {profile['age']} años")
        st.markdown(f"**Género:** {profile['gender']}")
        st.markdown(f"**Segmento:** {profile['segment']}")
    
    with col2:
        st.markdown(f"**Total Compras:** {profile['total_purchases']}")
        st.markdown(f"**Productos Únicos:** {profile['unique_items']}")
        st.markdown(f"**Gasto Total:** ${profile['total_spent']:.2f}")
    
    with col3:
        st.markdown(f"**Gasto Promedio:** ${profile['avg_spent']:.2f}")
        st.markdown(f"**Rating Promedio:** {profile['avg_rating']:.2f}/5")
        st.markdown(f"**Categoría Favorita:** {profile['favorite_category']}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent purchases
    st.markdown("**🛍️ Compras Recientes:**")
    for i, item in enumerate(profile['purchase_history'][:5], 1):
        st.markdown(f"{i}. {item}")

def display_recommendations_with_explanation(df, recommendations, explanation):
    """Display recommendations with detailed explanations"""
    
    # Algorithm explanation
    st.markdown('<div class="algorithm-box">', unsafe_allow_html=True)
    st.markdown(f"## 🎯 Recomendaciones usando SVD (Filtrado Colaborativo)")
    st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
    st.markdown(explanation)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not recommendations:
        st.warning("No se pudieron generar recomendaciones para este cliente")
        return
    
    st.markdown("### 📋 Productos Recomendados")
    
    # Create detailed recommendations table
    for i, (item, score) in enumerate(recommendations, 1):
        item_data = df[df['Item Purchased'] == item]
        if not item_data.empty:
            category = item_data['Category'].iloc[0]
            avg_price = item_data['Purchase Amount (USD)'].mean()
            avg_rating = item_data['Review Rating'].mean()
            popularity = len(item_data)
            
            st.markdown('<div class="recommendation-item">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{i}. {item}**")
                st.markdown(f"📦 Categoría: {category}")
                
            with col2:
                st.markdown(f"💰 **Precio:** ${avg_price:.2f}")
                st.markdown(f"⭐ **Rating:** {avg_rating:.1f}/5")
                
            with col3:
                st.markdown(f"🎯 **Score:** {score:.3f}")
                st.markdown(f"🔥 **Popularidad:** {popularity} compras")
            
            # Explanation for why this item is recommended
            if score > 4.0:
                confidence = "Alta"
                emoji = "🎯"
            elif score > 3.0:
                confidence = "Media"
                emoji = "👍"
            else:
                confidence = "Baja"
                emoji = "🤔"
                
            reason = f"Puntuación {confidence.lower()} ({score:.3f}) basada en patrones de usuarios similares"
            
            st.markdown(f"*{emoji} Confianza: {confidence} - {reason}*")
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application - Focused on SVD recommendations"""
    # Title
    st.markdown('<h1 class="main-header">🎯 Sistema de Recomendaciones SVD</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Análisis de Datos - Actividad 9<br>Filtrado Colaborativo usando Singular Value Decomposition</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Cargando datos y modelo SVD..."):
        df = load_data()
        svd_model = load_svd_model()
    
    if df is None:
        st.error("❌ Error al cargar los datos. Verifica que el archivo CSV esté en data/shopping_behavior_updated.csv")
        return
    
    # Introduction
    st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 📚 ¿Qué es el Filtrado Colaborativo con SVD?
    
    **SVD (Singular Value Decomposition)** es una técnica matemática que:
    - 🔍 **Encuentra patrones ocultos** en los datos de compras
    - 👥 **Identifica usuarios similares** basándose en sus preferencias
    - 🧮 **Factoriza la matriz** usuario-producto en componentes más simples
    - 🎯 **Predice preferencias** para productos no comprados
    
    **¿Por qué funciona?**
    - Si dos usuarios compraron productos similares en el pasado
    - Es probable que tengan gustos parecidos para productos futuros
    - SVD captura estas relaciones de manera automática y eficiente
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Total Clientes", f"{df['Customer ID'].nunique():,}")
    with col2:
        st.metric("📦 Total Productos", f"{df['Item Purchased'].nunique():,}")
    with col3:
        st.metric("🛒 Transacciones", f"{len(df):,}")
    with col4:
        st.metric("💰 Ingresos Totales", f"${df['Purchase Amount (USD)'].sum():,.0f}")
    
    st.markdown("---")
    
    # Customer selection
    st.markdown("## 🔍 Selecciona un Cliente para Analizar")
    
    customers = sorted(df['Customer ID'].unique())
    selected_customer = st.selectbox(
        "Elige un ID de cliente:", 
        customers,
        help="Cada cliente tiene un historial de compras único que influye en las recomendaciones"
    )
    
    # Number of recommendations
    num_recommendations = st.slider("Número de recomendaciones:", 3, 15, 8)
    
    if selected_customer:
        # Get and display customer profile
        profile = get_customer_profile(df, selected_customer)
        
        if profile:
            display_customer_profile(profile)
            
            st.markdown("---")
            
            # Generate recommendations button
            if st.button("🚀 Generar Recomendaciones SVD", type="primary", use_container_width=True):
                
                if svd_model is None:
                    st.error("❌ Modelo SVD no disponible. Verifica que el archivo svd_model.pkl esté en data/models/")
                    return
                    
                with st.spinner("🤖 Analizando patrones y generando recomendaciones..."):
                    recommendations, explanation = get_svd_recommendations(
                        svd_model, df, selected_customer, num_recommendations
                    )
                    
                    if recommendations:
                        display_recommendations_with_explanation(df, recommendations, explanation)
                        
                        # Additional insights
                        st.markdown("---")
                        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
                        st.markdown("""
                        ### 🧠 ¿Por qué estas recomendaciones?
                        
                        **El algoritmo SVD analizó:**
                        - 📊 **Patrones de compra** de todos los usuarios en el dataset
                        - 👥 **Usuarios similares** que compraron productos parecidos
                        - 🎯 **Preferencias latentes** (factores ocultos que influyen en las decisiones)
                        - ⚖️ **Puntuaciones predichas** para productos no comprados
                        
                        **Factores que influyen:**
                        - Historial de compras del cliente
                        - Comportamiento de usuarios con gustos similares
                        - Popularidad y calidad de los productos
                        - Patrones de precio y categoría
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Visualization of recommendations
                        if len(recommendations) > 0:
                            st.markdown("### 📊 Visualización de Puntuaciones")
                            
                            # Create bar chart of recommendation scores
                            items = [item for item, _ in recommendations]
                            scores = [score for _, score in recommendations]
                            
                            fig = px.bar(
                                x=scores,
                                y=[f"{i+1}. {item[:20]}..." if len(item) > 20 else f"{i+1}. {item}" 
                                   for i, item in enumerate(items)],
                                orientation='h',
                                title="Puntuaciones de Recomendación SVD",
                                labels={'x': 'Puntuación SVD', 'y': 'Productos Recomendados'},
                                color=scores,
                                color_continuous_scale='Viridis'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.warning("⚠️ No se pudieron generar recomendaciones para este cliente. Posibles razones:")
                        st.markdown("""
                        - El cliente puede no tener suficiente historial de compras
                        - Todos los productos disponibles ya fueron comprados
                        - Error en el modelo SVD
                        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p><strong>📚 Desarrollado para Análisis de Datos - Universidad</strong></p>
        <p>🎯 Este sistema demuestra el uso de SVD para sistemas de recomendación en e-commerce</p>
        <p>💡 <em>SVD es especialmente efectivo para datasets dispersos con muchos usuarios y productos</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()