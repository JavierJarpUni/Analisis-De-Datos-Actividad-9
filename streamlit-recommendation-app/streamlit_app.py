import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sistema de Recomendaciones - Actividad 9",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Generate synthetic e-commerce data for demonstration"""
    np.random.seed(42)
    
    # Product categories and items
    categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports']
    products = []
    
    for i in range(25):
        category = np.random.choice(categories)
        if category == 'Electronics':
            items = ['Smartphone', 'Laptop', 'Headphones', 'Tablet', 'Camera']
        elif category == 'Clothing':
            items = ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes']
        elif category == 'Books':
            items = ['Fiction Novel', 'Science Book', 'Biography', 'Cookbook', 'Manual']
        elif category == 'Home & Garden':
            items = ['Plant Pot', 'Lamp', 'Cushion', 'Tool Set', 'Decorative Item']
        else:  # Sports
            items = ['Running Shoes', 'Yoga Mat', 'Dumbbell', 'Sports Wear', 'Water Bottle']
        
        product_name = f"{np.random.choice(items)} {i+1}"
        products.append({
            'Product ID': f'P{i+1:03d}',
            'Product Name': product_name,
            'Category': category,
            'Price': np.random.uniform(10, 500),
            'Rating': np.random.uniform(3.0, 5.0)
        })
    
    # Generate customer data
    customers = []
    for i in range(100):
        customers.append({
            'Customer ID': f'C{i+1:03d}',
            'Age': np.random.randint(18, 70),
            'Gender': np.random.choice(['Male', 'Female']),
            'Location': np.random.choice(['Urban', 'Suburban', 'Rural']),
            'Purchase Power': np.random.choice(['Low', 'Medium', 'High'])
        })
    
    # Generate purchase history (ratings matrix)
    purchase_data = []
    for customer in customers:
        # Each customer rates 5-15 products
        num_purchases = np.random.randint(5, 16)
        purchased_products = np.random.choice(range(25), num_purchases, replace=False)
        
        for product_idx in purchased_products:
            rating = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])  # Bias towards higher ratings
            purchase_data.append({
                'Customer ID': customer['Customer ID'],
                'Product ID': f'P{product_idx+1:03d}',
                'Rating': rating,
                'Purchase Date': pd.date_range('2023-01-01', '2024-01-01', periods=1000)[np.random.randint(0, 1000)]
            })
    
    products_df = pd.DataFrame(products)
    customers_df = pd.DataFrame(customers)
    purchases_df = pd.DataFrame(purchase_data)
    
    return products_df, customers_df, purchases_df

@st.cache_data
def create_svd_model(purchases_df, n_components=10):
    """Create and train SVD model using TruncatedSVD"""
    # Create user-item matrix
    user_item_matrix = purchases_df.pivot_table(
        index='Customer ID', 
        columns='Product ID', 
        values='Rating', 
        fill_value=0
    )
    
    # Apply SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_
    
    # Reconstruct the matrix
    predicted_ratings = np.dot(user_factors, item_factors)
    predicted_df = pd.DataFrame(
        predicted_ratings,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    )
    
    return svd, user_item_matrix, predicted_df, user_factors, item_factors

def get_svd_recommendations(customer_id, predicted_df, user_item_matrix, products_df, top_n=5):
    """Get recommendations using SVD collaborative filtering"""
    if customer_id not in predicted_df.index:
        return pd.DataFrame()
    
    # Get customer's predicted ratings
    customer_ratings = predicted_df.loc[customer_id]
    
    # Get products the customer hasn't rated
    unrated_products = user_item_matrix.loc[customer_id][user_item_matrix.loc[customer_id] == 0].index
    
    # Get top recommendations from unrated products
    recommendations = customer_ratings[unrated_products].sort_values(ascending=False).head(top_n)
    
    # Merge with product details
    rec_df = pd.DataFrame({
        'Product ID': recommendations.index,
        'Predicted Rating': recommendations.values
    })
    
    result = rec_df.merge(products_df, on='Product ID', how='left')
    return result

def main():
    # Header
    st.markdown('<h1 class="main-header">🛍️ Sistema de Recomendaciones E-commerce</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Análisis de Datos - Actividad 9 | Filtrado Colaborativo con SVD</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Cargando datos y entrenando modelo...'):
        products_df, customers_df, purchases_df = load_sample_data()
        svd_model, user_item_matrix, predicted_df, user_factors, item_factors = create_svd_model(purchases_df)
    
    # Sidebar
    st.sidebar.header("🎯 Configuración")
    
    # Customer selection
    customer_ids = sorted(customers_df['Customer ID'].unique())
    selected_customer = st.sidebar.selectbox(
        "Seleccionar Cliente:",
        customer_ids,
        index=0
    )
    
    # Number of recommendations
    num_recommendations = st.sidebar.slider(
        "Número de Recomendaciones:",
        min_value=3,
        max_value=15,
        value=5
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Recomendaciones Personalizadas")
        
        # Get recommendations
        recommendations = get_svd_recommendations(
            selected_customer, 
            predicted_df, 
            user_item_matrix, 
            products_df, 
            num_recommendations
        )
        
        if not recommendations.empty:
            for idx, row in recommendations.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>🛍️ {row['Product Name']}</h4>
                        <p><strong>Categoría:</strong> {row['Category']}</p>
                        <p><strong>Precio:</strong> ${row['Price']:.2f}</p>
                        <p><strong>Rating Promedio:</strong> ⭐ {row['Rating']:.1f}</p>
                        <p><strong>Rating Predicho:</strong> 🎯 {row['Predicted Rating']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No se encontraron recomendaciones para este cliente.")
    
    with col2:
        st.header("👤 Perfil del Cliente")
        
        # Customer info
        customer_info = customers_df[customers_df['Customer ID'] == selected_customer].iloc[0]
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Información Personal</h4>
            <p><strong>ID:</strong> {customer_info['Customer ID']}</p>
            <p><strong>Edad:</strong> {customer_info['Age']} años</p>
            <p><strong>Género:</strong> {customer_info['Gender']}</p>
            <p><strong>Ubicación:</strong> {customer_info['Location']}</p>
            <p><strong>Poder Adquisitivo:</strong> {customer_info['Purchase Power']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Purchase history
        customer_purchases = purchases_df[purchases_df['Customer ID'] == selected_customer]
        
        if not customer_purchases.empty:
            st.subheader("📈 Historial de Compras")
            
            # Merge with product info
            purchase_history = customer_purchases.merge(products_df, on='Product ID', how='left')
            
            # Metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Productos", len(customer_purchases))
            with col_b:
                st.metric("Rating Promedio", f"{customer_purchases['Rating'].mean():.1f}")
            
            # Category distribution
            category_counts = purchase_history['Category'].value_counts()
            fig_cat = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Distribución por Categoría"
            )
            fig_cat.update_layout(height=300)
            st.plotly_chart(fig_cat, use_container_width=True)
            
            # Rating distribution
            rating_counts = customer_purchases['Rating'].value_counts().sort_index()
            fig_rating = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                title="Distribución de Ratings",
                labels={'x': 'Rating', 'y': 'Cantidad'}
            )
            fig_rating.update_layout(height=300)
            st.plotly_chart(fig_rating, use_container_width=True)
    
    # Analytics section
    st.header("📊 Análisis del Sistema")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Métricas del Modelo", "📈 Análisis de Datos", "ℹ️ Información Técnica"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Clientes", len(customers_df))
        with col2:
            st.metric("Productos", len(products_df))
        with col3:
            st.metric("Transacciones", len(purchases_df))
        with col4:
            density = (len(purchases_df) / (len(customers_df) * len(products_df))) * 100
            st.metric("Densidad Matriz", f"{density:.1f}%")
        
        # SVD Components visualization
        fig_components = go.Figure()
        fig_components.add_trace(go.Scatter(
            x=list(range(1, len(svd_model.explained_variance_ratio_) + 1)),
            y=svd_model.explained_variance_ratio_,
            mode='lines+markers',
            name='Varianza Explicada'
        ))
        fig_components.update_layout(
            title="Varianza Explicada por Componente SVD",
            xaxis_title="Componente",
            yaxis_title="Varianza Explicada",
            height=400
        )
        st.plotly_chart(fig_components, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Product category distribution
            category_dist = products_df['Category'].value_counts()
            fig1 = px.bar(
                x=category_dist.values,
                y=category_dist.index,
                orientation='h',
                title="Distribución de Productos por Categoría"
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Customer demographics
            age_groups = pd.cut(customers_df['Age'], bins=[0, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+'])
            age_dist = age_groups.value_counts()
            fig3 = px.pie(values=age_dist.values, names=age_dist.index, title="Distribución por Edad")
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Rating distribution
            rating_dist = purchases_df['Rating'].value_counts().sort_index()
            fig2 = px.bar(
                x=rating_dist.index,
                y=rating_dist.values,
                title="Distribución General de Ratings"
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Purchase power distribution
            power_dist = customers_df['Purchase Power'].value_counts()
            fig4 = px.pie(values=power_dist.values, names=power_dist.index, title="Poder Adquisitivo")
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        st.markdown("""
        ### 🔬 Información Técnica del Sistema
        
        **Algoritmo Implementado:**
        - **SVD (Singular Value Decomposition)** con scikit-learn TruncatedSVD
        - Factorización de matriz usuario-producto para filtrado colaborativo
        - Reducción de dimensionalidad para mejor rendimiento
        
        **Características del Dataset:**
        - Datos sintéticos generados para demostración
        - 100 clientes, 25 productos, ~1000 transacciones
        - 5 categorías de productos principales
        - Ratings del 3 al 5 (simulando preferencias positivas)
        
        **Métricas del Modelo:**
        - Componentes SVD: 10
        - Matriz de ratings dispersa con relleno de ceros
        - Recomendaciones basadas en productos no calificados por el usuario
        
        **Tecnologías Utilizadas:**
        - **Streamlit** para la interfaz web
        - **Pandas** para manipulación de datos
        - **NumPy** para operaciones numéricas
        - **Plotly** para visualizaciones interactivas
        - **Scikit-learn** para el modelo SVD
        
        **Autor:** Actividad 9 - Análisis de Datos
        """)

if __name__ == "__main__":
    main()