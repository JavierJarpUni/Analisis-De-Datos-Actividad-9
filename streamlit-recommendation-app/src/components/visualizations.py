import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizations:
    """Class for creating various visualizations"""
    
    def __init__(self, df):
        self.df = df
        
    def plot_customer_profile_metrics(self, profile):
        """Create customer profile visualization"""
        if not profile:
            st.error("No se pudo cargar el perfil del cliente")
            return
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Edad", f"{profile['age']} años")
            st.metric("Segmento", profile['segment'])
            
        with col2:
            st.metric("Compras Totales", profile['total_purchases'])
            st.metric("Productos Únicos", profile['unique_items'])
            
        with col3:
            st.metric("Gasto Total", f"${profile['total_spent']:.2f}")
            st.metric("Gasto Promedio", f"${profile['avg_spent']:.2f}")
            
        with col4:
            st.metric("Rating Promedio", f"{profile['avg_rating']:.2f}/5")
            st.metric("Categoría Favorita", profile['favorite_category'])
    
    def plot_recommendations_table(self, recommendations, title="Recomendaciones"):
        """Display recommendations in a nice table format"""
        if not recommendations:
            st.warning("No se encontraron recomendaciones")
            return
            
        st.subheader(title)
        
        # Create dataframe for recommendations
        rec_data = []
        for i, (item, score) in enumerate(recommendations, 1):
            item_data = self.df[self.df['Item Purchased'] == item]
            if not item_data.empty:
                category = item_data['Category'].iloc[0]
                avg_price = item_data['Purchase Amount (USD)'].mean()
                avg_rating = item_data['Review Rating'].mean()
                
                rec_data.append({
                    'Rank': i,
                    'Producto': item,
                    'Categoría': category,
                    'Precio Promedio': f"${avg_price:.2f}",
                    'Rating': f"{avg_rating:.2f}/5",
                    'Score': f"{score:.3f}"
                })
        
        if rec_data:
            rec_df = pd.DataFrame(rec_data)
            st.dataframe(rec_df, use_container_width=True)
    
    def plot_category_distribution(self):
        """Plot category distribution"""
        fig = px.bar(
            x=self.df['Category'].value_counts().values,
            y=self.df['Category'].value_counts().index,
            orientation='h',
            title="Distribución de Categorías",
            labels={'x': 'Número de Compras', 'y': 'Categoría'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_spending_by_segment(self):
        """Plot spending by customer segment"""
        segment_spending = self.df.groupby('Customer_Segment')['Purchase Amount (USD)'].agg(['mean', 'sum', 'count'])
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Gasto Promedio por Segmento', 'Número de Compras por Segmento']
        )
        
        # Average spending
        fig.add_trace(
            go.Bar(x=segment_spending.index, y=segment_spending['mean'], name='Gasto Promedio'),
            row=1, col=1
        )
        
        # Number of purchases
        fig.add_trace(
            go.Bar(x=segment_spending.index, y=segment_spending['count'], name='Número de Compras'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_top_products(self, category=None, top_n=10):
        """Plot top products overall or by category"""
        if category:
            data = self.df[self.df['Category'] == category]
            title = f"Top {top_n} Productos en {category}"
        else:
            data = self.df
            title = f"Top {top_n} Productos Más Vendidos"
        
        top_products = data['Item Purchased'].value_counts().head(top_n)
        
        fig = px.bar(
            x=top_products.values,
            y=top_products.index,
            orientation='h',
            title=title,
            labels={'x': 'Número de Compras', 'y': 'Producto'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_price_rating_scatter(self):
        """Plot price vs rating scatter plot"""
        # Aggregate data by product
        product_stats = self.df.groupby('Item Purchased').agg({
            'Purchase Amount (USD)': 'mean',
            'Review Rating': 'mean',
            'Category': 'first'
        }).reset_index()
        
        fig = px.scatter(
            product_stats,
            x='Purchase Amount (USD)',
            y='Review Rating',
            color='Category',
            title='Relación entre Precio y Rating de Productos',
            labels={'Purchase Amount (USD)': 'Precio Promedio (USD)', 'Review Rating': 'Rating Promedio'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_purchase_frequency_by_gender(self):
        """Plot purchase frequency by gender"""
        gender_stats = self.df.groupby(['Gender', 'Frequency of Purchases']).size().unstack(fill_value=0)
        
        fig = px.bar(
            gender_stats.T,
            title='Frecuencia de Compras por Género',
            labels={'index': 'Frecuencia de Compras', 'value': 'Número de Clientes'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_seasonal_trends(self):
        """Plot seasonal purchasing trends"""
        seasonal_data = self.df.groupby(['Season', 'Category']).size().unstack(fill_value=0)
        
        fig = px.bar(
            seasonal_data,
            title='Tendencias de Compra por Temporada',
            labels={'index': 'Temporada', 'value': 'Número de Compras'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_customer_lifetime_value_distribution(self):
        """Plot customer lifetime value distribution"""
        customer_value = self.df.groupby('Customer ID')['Purchase Amount (USD)'].sum()
        
        fig = px.histogram(
            x=customer_value.values,
            nbins=30,
            title='Distribución del Valor de Vida del Cliente',
            labels={'x': 'Valor Total de Compras (USD)', 'y': 'Número de Clientes'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)