import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    import config
except ImportError:
    # Fallback configuration
    class Config:
        SHOPPING_DATA_PATH = "../../data/shopping_behavior_updated.csv"
        SVD_MODEL_PATH = "../../data/models/svd_model.pkl"
        ITEM_SIMILARITY_PATH = "../../data/models/item_similarity_matrix.csv"
        RFM_ANALYSIS_PATH = "../../data/models/rfm_analysis.csv"
    
    config = Config()

class DataLoader:
    """Class to handle data loading and preprocessing"""
    
    def __init__(self):
        self.df = None
        self.user_item_matrix = None
        self.item_similarity_df = None
        self.rfm_df = None
        self.svd_model = None
        
    @st.cache_data
    def load_shopping_data(_self):
        """Load and preprocess shopping data"""
        try:
            df = pd.read_csv(config.SHOPPING_DATA_PATH)
            
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
            st.error(f"Error loading shopping data: {str(e)}")
            return None
    
    @st.cache_data
    def load_item_similarity(_self):
        """Load item similarity matrix"""
        try:
            return pd.read_csv(config.ITEM_SIMILARITY_PATH, index_col=0)
        except Exception as e:
            st.error(f"Error loading item similarity matrix: {str(e)}")
            return None
    
    @st.cache_data
    def load_rfm_analysis(_self):
        """Load RFM analysis data"""
        try:
            return pd.read_csv(config.RFM_ANALYSIS_PATH)
        except Exception as e:
            st.error(f"Error loading RFM analysis: {str(e)}")
            return None
    
    @st.cache_resource
    def load_svd_model(_self):
        """Load trained SVD model"""
        try:
            with open(config.SVD_MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading SVD model: {str(e)}")
            return None
    
    def create_user_item_matrix(self, df):
        """Create user-item matrix from dataframe"""
        try:
            user_item_matrix = df.pivot_table(
                index='Customer ID',
                columns='Item Purchased',
                values='interaction_score',
                aggfunc='mean'
            ).fillna(0)
            return user_item_matrix
        except Exception as e:
            st.error(f"Error creating user-item matrix: {str(e)}")
            return None
    
    def get_customer_data(self, df, customer_id):
        """Get data for a specific customer"""
        return df[df['Customer ID'] == customer_id]
    
    def get_unique_customers(self, df):
        """Get list of unique customer IDs"""
        return sorted(df['Customer ID'].unique())
    
    def get_unique_items(self, df):
        """Get list of unique items"""
        return sorted(df['Item Purchased'].unique())
    
    def get_categories(self, df):
        """Get list of unique categories"""
        return sorted(df['Category'].unique())