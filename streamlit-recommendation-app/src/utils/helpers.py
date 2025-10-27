import pandas as pd
import numpy as np
import streamlit as st
import base64
from datetime import datetime

def format_currency(amount):
    """Format number as currency"""
    return f"${amount:,.2f}"

def format_percentage(value):
    """Format number as percentage"""
    return f"{value:.2f}%"

def calculate_interaction_score(rating, amount, previous_purchases, max_amount, max_previous):
    """Calculate interaction score based on multiple factors"""
    rating_weight = 0.4
    amount_weight = 0.3
    loyalty_weight = 0.3
    
    normalized_amount = (amount / max_amount) * 5 if max_amount > 0 else 0
    normalized_loyalty = (previous_purchases / max_previous) * 5 if max_previous > 0 else 0
    
    interaction_score = (
        rating * rating_weight +
        normalized_amount * amount_weight +
        normalized_loyalty * loyalty_weight
    )
    
    return interaction_score

def get_age_segment(age):
    """Categorize age into segments"""
    if age <= 25:
        return 'Joven'
    elif age <= 40:
        return 'Adulto'
    elif age <= 60:
        return 'Maduro'
    else:
        return 'Senior'

def calculate_sparsity(matrix):
    """Calculate sparsity of user-item matrix"""
    total_elements = matrix.shape[0] * matrix.shape[1]
    non_zero_elements = (matrix > 0).sum().sum()
    sparsity = (1 - non_zero_elements / total_elements) * 100
    return sparsity

def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers, returning default if denominator is 0"""
    return numerator / denominator if denominator != 0 else default

def create_download_link(df, filename, link_text):
    """Create a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def validate_customer_id(customer_id, df):
    """Validate if customer ID exists in dataframe"""
    return customer_id in df['Customer ID'].unique()

def get_top_items_by_metric(df, metric_column, group_column=None, top_n=10):
    """Get top items by a specific metric"""
    if group_column:
        grouped = df.groupby([group_column, 'Item Purchased'])[metric_column].sum().reset_index()
        return grouped.nlargest(top_n, metric_column)
    else:
        return df.groupby('Item Purchased')[metric_column].sum().nlargest(top_n)

def calculate_customer_lifetime_value(customer_data):
    """Calculate customer lifetime value"""
    total_spent = customer_data['Purchase Amount (USD)'].sum()
    num_purchases = len(customer_data)
    avg_rating = customer_data['Review Rating'].mean()
    
    # Simple CLV calculation (can be made more sophisticated)
    clv = total_spent * (1 + avg_rating/5) * np.log(1 + num_purchases)
    return clv

def get_recommendation_explanation(item, customer_data, item_data):
    """Generate explanation for why an item was recommended"""
    explanations = []
    
    # Check category match
    customer_categories = customer_data['Category'].value_counts()
    item_category = item_data['Category'].iloc[0] if not item_data.empty else 'Unknown'
    
    if item_category in customer_categories.index:
        explanations.append(f"Te gusta la categoría {item_category}")
    
    # Check price range
    customer_avg_price = customer_data['Purchase Amount (USD)'].mean()
    item_price = item_data['Purchase Amount (USD)'].mean() if not item_data.empty else 0
    
    if abs(item_price - customer_avg_price) < customer_avg_price * 0.3:
        explanations.append("Precio similar a tus compras anteriores")
    
    # Check rating
    item_rating = item_data['Review Rating'].mean() if not item_data.empty else 0
    if item_rating >= 4.0:
        explanations.append("Producto bien calificado por otros usuarios")
    
    return explanations

@st.cache_data
def load_and_preprocess_data(file_path):
    """Load and preprocess data with caching"""
    try:
        df = pd.read_csv(file_path)
        
        # Basic preprocessing
        df['Purchase Amount (USD)'] = pd.to_numeric(df['Purchase Amount (USD)'], errors='coerce')
        df['Review Rating'] = pd.to_numeric(df['Review Rating'], errors='coerce')
        df['Previous Purchases'] = pd.to_numeric(df['Previous Purchases'], errors='coerce')
        
        # Remove rows with null values in critical columns
        df = df.dropna(subset=['Customer ID', 'Item Purchased', 'Purchase Amount (USD)', 'Review Rating'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_summary_stats(df):
    """Create summary statistics for the dataset"""
    stats = {
        'total_customers': df['Customer ID'].nunique(),
        'total_products': df['Item Purchased'].nunique(),
        'total_transactions': len(df),
        'total_revenue': df['Purchase Amount (USD)'].sum(),
        'avg_transaction_value': df['Purchase Amount (USD)'].mean(),
        'avg_rating': df['Review Rating'].mean(),
        'categories': df['Category'].nunique()
    }
    return stats

def generate_insights(df):
    """Generate business insights from the data"""
    insights = []
    
    # Top category
    top_category = df['Category'].value_counts().index[0]
    insights.append(f"La categoría más popular es '{top_category}'")
    
    # Customer segment analysis
    segment_analysis = df.groupby('Customer_Segment')['Purchase Amount (USD)'].mean()
    top_spending_segment = segment_analysis.idxmax()
    insights.append(f"El segmento '{top_spending_segment}' tiene el mayor gasto promedio")
    
    # Gender analysis
    gender_analysis = df.groupby('Gender')['Purchase Amount (USD)'].mean()
    if len(gender_analysis) > 1:
        top_gender = gender_analysis.idxmax()
        insights.append(f"Clientes '{top_gender}' gastan más en promedio")
    
    # Seasonal trends
    seasonal_analysis = df.groupby('Season')['Purchase Amount (USD)'].sum()
    top_season = seasonal_analysis.idxmax()
    insights.append(f"La temporada con mayores ventas es '{top_season}'")
    
    return insights