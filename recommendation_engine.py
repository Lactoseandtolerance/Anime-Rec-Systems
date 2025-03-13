import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import pickle
import logging
from datetime import datetime
from anime_data import AnimeData, UserData

class RecommendationEngine:
    """Class for generating anime recommendations using machine learning"""
    def __init__(self, anime_data, user_data):
        """Initialize with anime and user data objects"""
        self.anime_data = anime_data
        self.user_data = user_data
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models if they exist"""
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        model_files = {
            'content_model': 'models/content_model.pkl',
            'collaborative_model': 'models/collaborative_model.pkl',
            'kmeans_model': 'models/kmeans_model.pkl',
            'features': 'models/features.pkl'
        }
        
        for model_name, file_path in model_files.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    logging.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logging.error(f"Error loading model {model_name}: {e}")
    
    def save_models(self):
        """Save trained models to disk"""
        os.makedirs('models', exist_ok=True)
        
        for model_name, model in self.models.items():
            try:
                with open(f'models/{model_name}.pkl', 'wb') as f:
                    pickle.dump(model, f)
                logging.info(f"Saved model: {model_name}")
            except Exception as e:
                logging.error(f"Error saving model {model_name}: {e}")
    
    def extract_features(self, force_rebuild=False):
        """Extract features from anime data for content-based filtering"""
        if 'features' in self.models and not force_rebuild:
            logging.info("Using existing features")
            return self.models['features']
        
        logging.info("Extracting features for content-based filtering")
        
        # One-hot encode genres
        genre_features = pd.DataFrame(0, index=self.anime_data.df.index, columns=list(self.anime_data.all_genres))
        
        for idx, row in self.anime_data.df.iterrows():
            if isinstance(row['genre'], str):
                genres = [g.strip() for g in row['genre'].split(',')]
                for genre in genres:
                    if genre in genre_features.columns:
                        genre_features.loc[idx, genre] = 1
        
        # One-hot encode type
        type_encoder = OneHotEncoder(sparse_output=False)
        type_features = type_encoder.fit_transform(self.anime_data.df[['type']].fillna('Unknown'))
        type_df = pd.DataFrame(type_features, index=self.anime_data.df.index, 
                            columns=[f"type_{c}" for c in type_encoder.categories_[0]])
        
        # Normalize episode count and ratings
        scaler = StandardScaler()
        numeric_features = pd.DataFrame(
            scaler.fit_transform(self.anime_data.df[['episodes_num', 'rating', 'members']]),
            index=self.anime_data.df.index,
            columns=['episodes_scaled', 'rating_scaled', 'members_scaled']
        )
        
        # Combine all features
        features = pd.concat([genre_features, type_df, numeric_features], axis=1)
        
        # Apply dimensionality reduction if features are too many
        if features.shape[1] > 50:
            pca = PCA(n_components=min(50, features.shape[0], features.shape[1]))
            reduced_features = pca.fit_transform(features.fillna(0))
            features = pd.DataFrame(
                reduced_features, 
                index=self.anime_data.df.index,
                columns=[f'component_{i}' for i in range(reduced_features.shape[1])]
            )
            self.models['pca'] = pca
        
        self.models['features'] = features
        return features
    
    def train_content_based_model(self):
        """Train content-based recommendation model"""
        logging.info("Training content-based model")
        
        # Extract features
        features = self.extract_features()
        
        # Calculate similarity matrix
        similarity = cosine_similarity(features)
        self.models['content_model'] = similarity
        
        # Train a KMeans model for clustering similar anime
        kmeans = KMeans(n_clusters=10, random_state=42)
        self.anime_data.df['cluster'] = kmeans.fit_predict(features)
        self.models['kmeans_model'] = kmeans
        
        logging.info("Content-based model trained successfully")
        self.save_models()
        return similarity
    
    def train_collaborative_model(self):
        """Train collaborative filtering model"""
        logging.info("Training collaborative filtering model")
        
        if not self.user_data.user_ratings:
            logging.warning("Not enough user ratings for collaborative filtering")
            return None
        
        # Create user-item matrix (simplified for demo)
        user_items = np.zeros(len(self.anime_data.df))
        for anime_id, rating in self.user_data.user_ratings.items():
            try:
                idx = self.anime_data.df[self.anime_data.df['anime_id'] == int(anime_id)].index[0]
                user_items[idx] = rating
            except (IndexError, ValueError):
                continue
        
        # Train nearest neighbors model
        if sum(user_items > 0) >= 5:  # Need at least 5 ratings
            model = NearestNeighbors(metric='cosine', algorithm='brute')
            
            # Get anime features
            features = self.extract_features()
            
            # Weight features by user ratings
            weighted_features = features.copy()
            for i, rating in enumerate(user_items):
                if rating > 0:
                    weighted_features.iloc[i] = weighted_features.iloc[i] * (rating / 5.0)
            
            # Only use anime that the user has rated
            rated_indices = np.where(user_items > 0)[0]
            if len(rated_indices) >= 5:
                model.fit(weighted_features.iloc[rated_indices])
                self.models['collaborative_model'] = {
                    'model': model,
                    'indices': rated_indices
                }
                logging.info("Collaborative model trained successfully")
                self.save_models()
                return model
        
        logging.warning("Not enough data to train collaborative model")
        return None
    
    def get_content_recommendations(self, anime_id, top_n=10):
        """Get content-based recommendations based on item similarity"""
        try:
            if 'content_model' not in self.models:
                self.train_content_based_model()
            
            similarity = self.models['content_model']
            idx = self.anime_data.df[self.anime_data.df['anime_id'] == anime_id].index[0]
            
            # Get similarity scores and sort by similarity
            sim_scores = list(enumerate(similarity[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top N most similar anime (excluding the input anime)
            sim_scores = sim_scores[1:top_n+1]
            anime_indices = [i[0] for i in sim_scores]
            
            # Record recommendations in history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.user_data.recommendation_history.append({
                'timestamp': timestamp,
                'method': 'content',
                'input_anime': anime_id,
                'recommendations': self.anime_data.df.iloc[anime_indices]['anime_id'].tolist()
            })
            self.user_data.save_user_data()
            
            return self.anime_data.df.iloc[anime_indices][['anime_id', 'name', 'genre', 'type', 'episodes', 'rating']]
        except Exception as e:
            logging.error(f"Error in content recommendations: {e}")
            return pd.DataFrame()
    
    def get_collaborative_recommendations(self, top_n=10):
        """Get collaborative filtering recommendations based on user profile"""
        try:
            if 'collaborative_model' not in self.models or not self.user_data.user_ratings:
                self.train_collaborative_model()
            
            if 'collaborative_model' not in self.models:
                return pd.DataFrame()  # Not enough data
            
            # Get feature vector of user profile
            features = self.extract_features()
            user_vector = np.zeros(features.shape[1])
            
            # Create a weighted average of features based on user's ratings
            total_weight = 0
            for anime_id, rating in self.user_data.user_ratings.items():
                try:
                    idx = self.anime_data.df[self.anime_data.df['anime_id'] == int(anime_id)].index[0]
                    weight = rating / 5.0  # Normalize to 0-1
                    user_vector += features.iloc[idx].values * weight
                    total_weight += weight
                except (IndexError, ValueError):
                    continue
            
            if total_weight > 0:
                user_vector /= total_weight
            
            # Get recommendations using nearest neighbors
            model = self.models['collaborative_model']['model']
            distances, indices = model.kneighbors([user_vector], n_neighbors=top_n)
            
            # Map back to original indices
            orig_indices = self.models['collaborative_model']['indices'][indices[0]]
            
            # Filter out anime that the user has already rated
            rated_anime_ids = set(int(id) for id in self.user_data.user_ratings.keys())
            recommendations = self.anime_data.df.iloc[orig_indices]
            recommendations = recommendations[~recommendations['anime_id'].isin(rated_anime_ids)]
            
            # Record recommendations in history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.user_data.recommendation_history.append({
                'timestamp': timestamp,
                'method': 'collaborative',
                'input_anime': None,
                'recommendations': recommendations['anime_id'].tolist()
            })
            self.user_data.save_user_data()
            
            return recommendations[['anime_id', 'name', 'genre', 'type', 'episodes', 'rating']].head(top_n)
        except Exception as e:
            logging.error(f"Error in collaborative recommendations: {e}")
            return pd.DataFrame()
    
    def get_hybrid_recommendations(self, anime_id=None, top_n=10):
        """Get hybrid recommendations combining content and collaborative filtering"""
        try:
            content_df = pd.DataFrame()
            collab_df = pd.DataFrame()
            
            # Get content-based recommendations if anime_id is provided
            if anime_id is not None:
                content_df = self.get_content_recommendations(anime_id, top_n=top_n)
            
            # Get collaborative filtering recommendations if possible
            if self.user_data.user_ratings and len(self.user_data.user_ratings) >= 5:
                collab_df = self.get_collaborative_recommendations(top_n=top_n)
            
            # If only one method returns results, use those
            if content_df.empty and not collab_df.empty:
                return collab_df
            elif not content_df.empty and collab_df.empty:
                return content_df
            elif content_df.empty and collab_df.empty:
                # Fallback: return top-rated anime
                return self.anime_data.df.sort_values('rating', ascending=False).head(top_n)[
                    ['anime_id', 'name', 'genre', 'type', 'episodes', 'rating']]
            
            # Combine results (50% content, 50% collaborative)
            content_weight = 0.5
            collaborative_weight = 0.5
            
            # Calculate a combined score
            combined_results = {}
            
            # Add content recommendations with weight
            for _, row in content_df.iterrows():
                anime_id = row['anime_id']
                combined_results[anime_id] = {
                    'score': content_weight,
                    'row': row
                }
            
            # Add collaborative recommendations with weight
            for _, row in collab_df.iterrows():
                anime_id = row['anime_id']
                if anime_id in combined_results:
                    combined_results[anime_id]['score'] += collaborative_weight
                else:
                    combined_results[anime_id] = {
                        'score': collaborative_weight,
                        'row': row
                    }
            
            # Sort by combined score and return top_n
            sorted_results = sorted(combined_results.items(), 
                                   key=lambda x: x[1]['score'], 
                                   reverse=True)[:top_n]
            
            # Create DataFrame from results
            result_rows = [item[1]['row'] for item in sorted_results]
            
            # Record recommendations in history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.user_data.recommendation_history.append({
                'timestamp': timestamp,
                'method': 'hybrid',
                'input_anime': anime_id,
                'recommendations': [int(item[0]) for item in sorted_results]
            })
            self.user_data.save_user_data()
            
            return pd.DataFrame(result_rows)
        except Exception as e:
            logging.error(f"Error in hybrid recommendations: {e}")
            return pd.DataFrame()