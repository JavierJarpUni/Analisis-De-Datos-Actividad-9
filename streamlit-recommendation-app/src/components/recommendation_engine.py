import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

class RecommendationEngine:
    """Main recommendation engine class"""
    
    def __init__(self, df, svd_model, item_similarity_df):
        self.df = df
        self.svd_model = svd_model
        self.item_similarity_df = item_similarity_df
        self.user_item_matrix = self.create_user_item_matrix()
        
    def create_user_item_matrix(self):
        """Create user-item matrix"""
        return self.df.pivot_table(
            index='Customer ID',
            columns='Item Purchased',
            values='interaction_score',
            aggfunc='mean'
        ).fillna(0)
    
    def get_item_based_recommendations(self, item_name, top_n=5):
        """Get item-based collaborative filtering recommendations"""
        if item_name not in self.item_similarity_df.index:
            return []
        
        similar_items = self.item_similarity_df[item_name].sort_values(ascending=False)[1:top_n+1]
        return list(zip(similar_items.index, similar_items.values))
    
    def get_svd_recommendations(self, customer_id, top_n=5):
        """Get SVD-based recommendations"""
        if self.svd_model is None:
            return []
            
        # Get all items
        all_items = self.df['Item Purchased'].unique()
        
        # Get items already purchased by customer
        customer_items = set(self.df[self.df['Customer ID'] == customer_id]['Item Purchased'])
        
        # Predict scores for unpurchased items
        predictions_list = []
        
        for item in all_items:
            if item not in customer_items:
                pred = self.svd_model.predict(customer_id, item)
                predictions_list.append((item, pred.est))
        
        # Sort and return top N
        predictions_list.sort(key=lambda x: x[1], reverse=True)
        return predictions_list[:top_n]
    
    def get_hybrid_recommendations(self, customer_id, top_n=5, alpha=0.6):
        """Get hybrid recommendations (collaborative + content-based)"""
        # Get collaborative recommendations
        collab_recs = self.get_svd_recommendations(customer_id, top_n=top_n*2)
        
        if not collab_recs:
            return []
        
        # Get user preferences
        customer_data = self.df[self.df['Customer ID'] == customer_id]
        if customer_data.empty:
            return collab_recs[:top_n]
        
        # Get user's favorite category
        favorite_category = customer_data['Category'].mode().iloc[0] if len(customer_data['Category'].mode()) > 0 else None
        
        # Calculate hybrid scores
        hybrid_scores = []
        for item, collab_score in collab_recs:
            item_data = self.df[self.df['Item Purchased'] == item]
            if not item_data.empty:
                item_category = item_data['Category'].iloc[0]
                
                # Content score based on category match
                content_score = 5.0 if item_category == favorite_category else 2.5
                
                # Combine scores
                hybrid_score = alpha * collab_score + (1 - alpha) * content_score
                hybrid_scores.append((item, hybrid_score))
        
        # Sort and return
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores[:top_n]
    
    def get_popular_items_by_category(self, category, top_n=10):
        """Get most popular items in a category"""
        category_data = self.df[self.df['Category'] == category]
        popular_items = category_data['Item Purchased'].value_counts().head(top_n)
        
        results = []
        for item, count in popular_items.items():
            item_data = category_data[category_data['Item Purchased'] == item]
            avg_price = item_data['Purchase Amount (USD)'].mean()
            avg_rating = item_data['Review Rating'].mean()
            results.append({
                'item': item,
                'purchases': count,
                'avg_price': avg_price,
                'avg_rating': avg_rating
            })
        
        return results
    
    def get_customer_profile(self, customer_id):
        """Get comprehensive customer profile"""
        customer_data = self.df[self.df['Customer ID'] == customer_id]
        
        if customer_data.empty:
            return None
        
        profile = {
            'customer_id': customer_id,
            'age': customer_data['Age'].iloc[0],
            'gender': customer_data['Gender'].iloc[0],
            'segment': customer_data['Customer_Segment'].iloc[0],
            'location': customer_data['Location'].iloc[0] if 'Location' in customer_data.columns else 'N/A',
            'total_purchases': len(customer_data),
            'unique_items': customer_data['Item Purchased'].nunique(),
            'total_spent': customer_data['Purchase Amount (USD)'].sum(),
            'avg_spent': customer_data['Purchase Amount (USD)'].mean(),
            'avg_rating': customer_data['Review Rating'].mean(),
            'favorite_category': customer_data['Category'].mode().iloc[0] if len(customer_data['Category'].mode()) > 0 else 'N/A',
            'purchase_history': customer_data['Item Purchased'].unique()[:5]  # Last 5 items
        }
        
        return profile