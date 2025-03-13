import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='anime_recommender.log')

class AnimeRecommendationSystem:
    def __init__(self, csv_path="anime.csv"):
        """Initialize the recommendation system with the dataset"""
        self.data_path = csv_path
        self.load_data()
        self.user_profile = {}
        self.user_ratings = {}
        self.feedback_history = []
        self.recommendation_history = []
        self.models = {}
        
        # Load saved models and user data if they exist
        self.load_models()
        self.load_user_data()
        
    def load_data(self):
        """Load and preprocess the anime dataset"""
        try:
            logging.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            
            # Handle missing values
            self.df['rating'] = self.df['rating'].fillna(self.df['rating'].mean())
            self.df['genre'] = self.df['genre'].fillna("Unknown")
            self.df['type'] = self.df['type'].fillna("Unknown")
            self.df['episodes'] = self.df['episodes'].fillna("Unknown")
            
            # Create a numeric episodes column for analysis
            self.df['episodes_num'] = pd.to_numeric(self.df['episodes'], errors='coerce')
            self.df['episodes_num'] = self.df['episodes_num'].fillna(0)
            
            # Prepare genre data for feature extraction
            self.all_genres = set()
            for genres in self.df['genre'].str.split(','):
                if isinstance(genres, list):
                    for genre in genres:
                        self.all_genres.add(genre.strip())
            
            logging.info(f"Data loaded successfully. {len(self.df)} anime entries found.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise
    
    def load_models(self):
        """Load pre-trained models if they exist"""
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
    
    def load_user_data(self):
        """Load user profile and ratings if they exist"""
        if os.path.exists('user_data.json'):
            try:
                with open('user_data.json', 'r') as f:
                    user_data = json.load(f)
                    self.user_profile = user_data.get('profile', {})
                    self.user_ratings = user_data.get('ratings', {})
                    self.feedback_history = user_data.get('feedback_history', [])
                    self.recommendation_history = user_data.get('recommendation_history', [])
                logging.info("Loaded user data")
            except Exception as e:
                logging.error(f"Error loading user data: {e}")
    
    def save_user_data(self):
        """Save user profile and ratings to disk"""
        user_data = {
            'profile': self.user_profile,
            'ratings': self.user_ratings,
            'feedback_history': self.feedback_history,
            'recommendation_history': self.recommendation_history
        }
        
        try:
            with open('user_data.json', 'w') as f:
                json.dump(user_data, f)
            logging.info("Saved user data")
        except Exception as e:
            logging.error(f"Error saving user data: {e}")
    
    def extract_features(self, force_rebuild=False):
        """Extract features from anime data for content-based filtering"""
        if 'features' in self.models and not force_rebuild:
            logging.info("Using existing features")
            return self.models['features']
        
        logging.info("Extracting features for content-based filtering")
        
        # One-hot encode genres
        genre_features = pd.DataFrame(0, index=self.df.index, columns=list(self.all_genres))
        
        for idx, row in self.df.iterrows():
            if isinstance(row['genre'], str):
                genres = [g.strip() for g in row['genre'].split(',')]
                for genre in genres:
                    if genre in genre_features.columns:
                        genre_features.loc[idx, genre] = 1
        
        # One-hot encode type
        type_encoder = OneHotEncoder(sparse=False)
        type_features = type_encoder.fit_transform(self.df[['type']])
        type_df = pd.DataFrame(type_features, index=self.df.index, 
                               columns=[f"type_{c}" for c in type_encoder.categories_[0]])
        
        # Normalize episode count and ratings
        scaler = StandardScaler()
        numeric_features = pd.DataFrame(
            scaler.fit_transform(self.df[['episodes_num', 'rating', 'members']]),
            index=self.df.index,
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
                index=self.df.index,
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
        self.df['cluster'] = kmeans.fit_predict(features)
        self.models['kmeans_model'] = kmeans
        
        logging.info("Content-based model trained successfully")
        return similarity
    
    def train_collaborative_model(self):
        """Train collaborative filtering model"""
        logging.info("Training collaborative filtering model")
        
        if not self.user_ratings:
            logging.warning("Not enough user ratings for collaborative filtering")
            return None
        
        # Create user-item matrix (simplified for demo)
        # In a real system, you'd have ratings from multiple users
        user_items = np.zeros(len(self.df))
        for anime_id, rating in self.user_ratings.items():
            try:
                idx = self.df[self.df['anime_id'] == int(anime_id)].index[0]
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
                return model
        
        logging.warning("Not enough data to train collaborative model")
        return None
    
    def get_content_recommendations(self, anime_id, top_n=10):
        """Get content-based recommendations based on item similarity"""
        try:
            if 'content_model' not in self.models:
                self.train_content_based_model()
            
            similarity = self.models['content_model']
            idx = self.df[self.df['anime_id'] == anime_id].index[0]
            
            # Get similarity scores and sort by similarity
            sim_scores = list(enumerate(similarity[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top N most similar anime (excluding the input anime)
            sim_scores = sim_scores[1:top_n+1]
            anime_indices = [i[0] for i in sim_scores]
            
            # Record recommendations in history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.recommendation_history.append({
                'timestamp': timestamp,
                'method': 'content',
                'input_anime': anime_id,
                'recommendations': self.df.iloc[anime_indices]['anime_id'].tolist()
            })
            
            return self.df.iloc[anime_indices][['anime_id', 'name', 'genre', 'type', 'episodes', 'rating']]
        except Exception as e:
            logging.error(f"Error in content recommendations: {e}")
            return pd.DataFrame()
    
    def get_collaborative_recommendations(self, top_n=10):
        """Get collaborative filtering recommendations based on user profile"""
        try:
            if 'collaborative_model' not in self.models or not self.user_ratings:
                self.train_collaborative_model()
            
            if 'collaborative_model' not in self.models:
                return pd.DataFrame()  # Not enough data
            
            # Get feature vector of user profile
            features = self.extract_features()
            user_vector = np.zeros(features.shape[1])
            
            # Create a weighted average of features based on user's ratings
            total_weight = 0
            for anime_id, rating in self.user_ratings.items():
                try:
                    idx = self.df[self.df['anime_id'] == int(anime_id)].index[0]
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
            rated_anime_ids = set(int(id) for id in self.user_ratings.keys())
            recommendations = self.df.iloc[orig_indices]
            recommendations = recommendations[~recommendations['anime_id'].isin(rated_anime_ids)]
            
            # Record recommendations in history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.recommendation_history.append({
                'timestamp': timestamp,
                'method': 'collaborative',
                'input_anime': None,
                'recommendations': recommendations['anime_id'].tolist()
            })
            
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
            if self.user_ratings and len(self.user_ratings) >= 5:
                collab_df = self.get_collaborative_recommendations(top_n=top_n)
            
            # If only one method returns results, use those
            if content_df.empty and not collab_df.empty:
                return collab_df
            elif not content_df.empty and collab_df.empty:
                return content_df
            elif content_df.empty and collab_df.empty:
                # Fallback: return top-rated anime
                return self.df.sort_values('rating', ascending=False).head(top_n)[
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
            self.recommendation_history.append({
                'timestamp': timestamp,
                'method': 'hybrid',
                'input_anime': anime_id,
                'recommendations': [int(item[0]) for item in sorted_results]
            })
            
            return pd.DataFrame(result_rows)
        except Exception as e:
            logging.error(f"Error in hybrid recommendations: {e}")
            return pd.DataFrame()
    
    def record_rating(self, anime_id, rating):
        """Record user rating for an anime"""
        try:
            anime_id = int(anime_id)
            rating = float(rating)
            if 1 <= rating <= 10:
                self.user_ratings[str(anime_id)] = rating
                logging.info(f"Recorded rating {rating} for anime {anime_id}")
                
                # Retrain collaborative model if we have enough ratings
                if len(self.user_ratings) >= 5:
                    self.train_collaborative_model()
                
                self.save_user_data()
                return True
            else:
                logging.warning(f"Invalid rating value: {rating}")
                return False
        except ValueError as e:
            logging.error(f"Error recording rating: {e}")
            return False
    
    def record_feedback(self, anime_id, liked=True, comment=""):
        """Record user feedback on a recommendation"""
        try:
            feedback = {
                'anime_id': int(anime_id),
                'liked': liked,
                'comment': comment,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.feedback_history.append(feedback)
            logging.info(f"Recorded feedback for anime {anime_id}: {'liked' if liked else 'disliked'}")
            
            # Use feedback to adjust recommendations
            # This is a placeholder for more sophisticated reinforcement learning
            if liked:
                # Increase weight for this anime's features in future recommendations
                # For simplicity, we just record a positive rating
                if str(anime_id) not in self.user_ratings:
                    self.user_ratings[str(anime_id)] = 8.0  # Assume they liked it ~8/10
            else:
                # Decrease weight for this anime's features
                # For simplicity, just record a low rating
                if str(anime_id) not in self.user_ratings:
                    self.user_ratings[str(anime_id)] = 3.0  # Assume they didn't like it ~3/10
            
            self.save_user_data()
            return True
        except Exception as e:
            logging.error(f"Error recording feedback: {e}")
            return False
    
    def update_user_profile(self, preferences):
        """Update user profile with genre and type preferences"""
        try:
            self.user_profile.update(preferences)
            logging.info(f"Updated user profile: {preferences}")
            self.save_user_data()
            return True
        except Exception as e:
            logging.error(f"Error updating user profile: {e}")
            return False
    
    def get_genre_statistics(self):
        """Get statistics about anime genres"""
        genre_counts = {}
        for genres in self.df['genre'].str.split(','):
            if isinstance(genres, list):
                for genre in genres:
                    genre = genre.strip()
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        return sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    
    def get_rating_statistics(self):
        """Get statistics about anime ratings"""
        return {
            'mean': self.df['rating'].mean(),
            'median': self.df['rating'].median(),
            'min': self.df['rating'].min(),
            'max': self.df['rating'].max(),
            'distribution': np.histogram(self.df['rating'].dropna(), bins=10)
        }
    
    def search_anime(self, criteria):
        """Search for anime based on various criteria"""
        query = self.df.copy()
        
        # Filter by name
        if 'name' in criteria and criteria['name']:
            query = query[query['name'].str.contains(criteria['name'], case=False, na=False)]
        
        # Filter by genre
        if 'genre' in criteria and criteria['genre']:
            genres = [g.strip() for g in criteria['genre'].split(',')]
            for genre in genres:
                query = query[query['genre'].str.contains(genre, case=False, na=False)]
        
        # Filter by type
        if 'type' in criteria and criteria['type']:
            query = query[query['type'] == criteria['type']]
        
        # Filter by minimum rating
        if 'min_rating' in criteria and criteria['min_rating']:
            query = query[query['rating'] >= float(criteria['min_rating'])]
        
        # Filter by maximum rating
        if 'max_rating' in criteria and criteria['max_rating']:
            query = query[query['rating'] <= float(criteria['max_rating'])]
        
        # Filter by episodes
        if 'min_episodes' in criteria and criteria['min_episodes']:
            query = query[pd.to_numeric(query['episodes'], errors='coerce') >= int(criteria['min_episodes'])]
        
        if 'max_episodes' in criteria and criteria['max_episodes']:
            query = query[pd.to_numeric(query['episodes'], errors='coerce') <= int(criteria['max_episodes'])]
        
        # Sort results
        if 'sort_by' in criteria and criteria['sort_by']:
            sort_field = criteria['sort_by']
            ascending = True
            if sort_field == 'rating':
                ascending = False
            query = query.sort_values(sort_field, ascending=ascending)
        
        return query[['anime_id', 'name', 'genre', 'type', 'episodes', 'rating']]


class AnimeRecommenderGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Advanced Anime Recommendation System")
        self.master.geometry("1200x800")
        self.master.minsize(900, 600)
        
        # Create recommender engine
        self.recommender = AnimeRecommendationSystem()
        
        # Create tabbed interface
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.search_tab = ttk.Frame(self.notebook)
        self.recommendations_tab = ttk.Frame(self.notebook)
        self.profile_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.search_tab, text="Search Anime")
        self.notebook.add(self.recommendations_tab, text="Get Recommendations")
        self.notebook.add(self.profile_tab, text="User Profile")
        self.notebook.add(self.stats_tab, text="Analytics")
        
        # Set up each tab
        self.setup_search_tab()
        self.setup_recommendations_tab()
        self.setup_profile_tab()
        self.setup_stats_tab()
        
        # Initialize recommender models
        self.init_models()
    
    def init_models(self):
        """Initialize recommendation models in a background process"""
        # Train models if they don't exist
        if 'content_model' not in self.recommender.models:
            self.master.after(100, self.recommender.train_content_based_model)
        
        if len(self.recommender.user_ratings) >= 5 and 'collaborative_model' not in self.recommender.models:
            self.master.after(200, self.recommender.train_collaborative_model)
    
    def setup_search_tab(self):
        """Set up the search interface"""
        frame = ttk.Frame(self.search_tab, padding="10")
        frame.pack(fill='both', expand=True)
        
        # Search criteria
        criteria_frame = ttk.LabelFrame(frame, text="Search Criteria", padding="10")
        criteria_frame.pack(fill='x', pady=10)
        
        # Name search
        ttk.Label(criteria_frame, text="Anime Name:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.name_entry = ttk.Entry(criteria_frame, width=30)
        self.name_entry.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        # Genre search
        ttk.Label(criteria_frame, text="Genre(s):").grid(row=0, column=2, sticky='w', padx=5, pady=5)
        self.genre_entry = ttk.Entry(criteria_frame, width=30)
        self.genre_entry.grid(row=0, column=3, sticky='w', padx=5, pady=5)
        
        # Genre browse button
        ttk.Button(criteria_frame, text="Browse Genres", command=self.browse_genres).grid(
            row=0, column=4, sticky='w', padx=5, pady=5)
        
        # Type filter
        ttk.Label(criteria_frame, text="Type:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.type_var = tk.StringVar()
        type_options = ["", "TV", "Movie", "OVA", "Special", "ONA", "Music"]
        self.type_combo = ttk.Combobox(criteria_frame, textvariable=self.type_var, values=type_options, width=10)
        self.type_combo.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        # Rating range
        ttk.Label(criteria_frame, text="Min Rating:").grid(row=1, column=2, sticky='w', padx=5, pady=5)
        self.min_rating_var = tk.StringVar()
        self.min_rating_combo = ttk.Combobox(criteria_frame, textvariable=self.min_rating_var, 
                                              values=[""] + [str(i) for i in range(1, 11)], width=5)
        self.min_rating_combo.grid(row=1, column=3, sticky='w', padx=5, pady=5)
        
        ttk.Label(criteria_frame, text="Max Rating:").grid(row=1, column=4, sticky='w', padx=5, pady=5)
        self.max_rating_var = tk.StringVar()
        self.max_rating_combo = ttk.Combobox(criteria_frame, textvariable=self.max_rating_var, 
                                              values=[""] + [str(i) for i in range(1, 11)], width=5)
        self.max_rating_combo.grid(row=2, column=0, sticky='w', padx=5, pady=5)
        
        # Episodes range
        ttk.Label(criteria_frame, text="Min Episodes:").grid(row=2, column=1, sticky='w', padx=5, pady=5)
        self.min_episodes_var = tk.StringVar()
        self.min_episodes_entry = ttk.Entry(criteria_frame, textvariable=self.min_episodes_var, width=5)
        self.min_episodes_entry.grid(row=2, column=2, sticky='w', padx=5, pady=5)
        
        ttk.Label(criteria_frame, text="Max Episodes:").grid(row=2, column=3, sticky='w', padx=5, pady=5)
        self.max_episodes_var = tk.StringVar()
        self.max_episodes_entry = ttk.Entry(criteria_frame, textvariable=self.max_episodes_var, width=5)
        self.max_episodes_entry.grid(row=2, column=4, sticky='w', padx=5, pady=5)
        
        # Sort options
        ttk.Label(criteria_frame, text="Sort By:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.sort_var = tk.StringVar()
        sort_options = ["name", "rating", "members"]
        self.sort_combo = ttk.Combobox(criteria_frame, textvariable=self.sort_var, values=sort_options, width=10)
        self.sort_combo.grid(row=3, column=1, sticky='w', padx=5, pady=5)
        
        # Search button
        search_btn = ttk.Button(criteria_frame, text="Search", command=self.perform_search)
        search_btn.grid(row=3, column=3, padx=5, pady=5)
        
        # Clear button
        clear_btn = ttk.Button(criteria_frame, text="Clear", command=self.clear_search)
        clear_btn.grid(row=3, column=4, padx=5, pady=5)
        
        # Results area
        results_frame = ttk.LabelFrame(frame, text="Search Results", padding="10")
        results_frame.pack(fill='both', expand=True, pady=10)
        
        # Treeview for displaying results
        self.results_tree = ttk.Treeview(results_frame, columns=("ID", "Name", "Genre", "Type", "Episodes", "Rating"))
        self.results_tree.pack(side='left', fill='both', expand=True)
        
        # Configure tree columns
        self.results_tree.column("#0", width=0, stretch=tk.NO)  # Hide first column
        self.results_tree.column("ID", width=50, anchor=tk.CENTER)
        self.results_tree.column("Name", width=200)
        self.results_tree.column("Genre", width=200)
        self.results_tree.column("Type", width=80, anchor=tk.CENTER)
        self.results_tree.column("Episodes", width=80, anchor=tk.CENTER)
        self.results_tree.column("Rating", width=80, anchor=tk.CENTER)
        
        # Set column headings
        self.results_tree.heading("ID", text="ID")
        self.results_tree.heading("Name", text="Name")
        self.results_tree.heading("Genre", text="Genre")
        self.results_tree.heading("Type", text="Type")
        self.results_tree.heading("Episodes", text="Episodes")
        self.results_tree.heading("Rating", text="Rating")
        
        # Add scrollbar
        results_scroll = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        results_scroll.pack(side='right', fill='y')
        self.results_tree.configure(yscrollcommand=results_scroll.set)
        
        # Right-click menu for tree
        self.tree_menu = tk.Menu(self.master, tearoff=0)
        self.tree_menu.add_command(label="Recommend Similar Anime", command=self.recommend_similar)
        self.tree_menu.add_command(label="Rate This Anime", command=self.rate_anime)
        self.tree_menu.add_separator()
        self.tree_menu.add_command(label="View Details", command=self.view_anime_details)
        
        # Bind right-click event
        self.results_tree.bind("<Button-3>", self.show_tree_menu)
        self.results_tree.bind("<Double-1>", lambda e: self.view_anime_details())
    
    def setup_recommendations_tab(self):
        """Set up the recommendations interface"""
        frame = ttk.Frame(self.recommendations_tab, padding="10")
        frame.pack(fill='both', expand=True)
        
        # Recommendation options
        options_frame = ttk.LabelFrame(frame, text="Recommendation Options", padding="10")
        options_frame.pack(fill='x', pady=10)
        
        # Recommendation type
        ttk.Label(options_frame, text="Recommendation Type:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.rec_type_var = tk.StringVar(value="hybrid")
        rec_types = [
            ("Hybrid (Content + Collaborative)", "hybrid"),
            ("Content-Based (Similar Anime)", "content"),
            ("Collaborative (Based on Ratings)", "collaborative")
        ]
        
        for i, (text, value) in enumerate(rec_types):
            ttk.Radiobutton(options_frame, text=text, variable=self.rec_type_var, value=value).grid(
                row=0, column=i+1, sticky='w', padx=5, pady=5)
        
        # Base anime for content recommendations
        ttk.Label(options_frame, text="Based on Anime:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.base_anime_entry = ttk.Entry(options_frame, width=30)
        self.base_anime_entry.grid(row=1, column=1, columnspan=2, sticky='w', padx=5, pady=5)
        
        ttk.Button(options_frame, text="Browse Anime", command=self.browse_anime).grid(
            row=1, column=3, sticky='w', padx=5, pady=5)
        
        # Number of recommendations
        ttk.Label(options_frame, text="Number of Recommendations:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.num_recs_var = tk.StringVar(value="10")
        num_options = ["5", "10", "15", "20", "25"]
        self.num_recs_combo = ttk.Combobox(options_frame, textvariable=self.num_recs_var, values=num_options, width=5)
        self.num_recs_combo.grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        # Get recommendations button
        get_rec_btn = ttk.Button(options_frame, text="Get Recommendations", command=self.get_recommendations)
        get_rec_btn.grid(row=2, column=3, padx=5, pady=5)
        
        # Recommendations results
        rec_results_frame = ttk.LabelFrame(frame, text="Recommendations", padding="10")
        rec_results_frame.pack(fill='both', expand=True, pady=10)
        
        # Treeview for displaying recommendations
        self.rec_tree = ttk.Treeview(rec_results_frame, columns=("ID", "Name", "Genre", "Type", "Episodes", "Rating"))
        self.rec_tree.pack(side='left', fill='both', expand=True)
        
        # Configure tree columns (same as search results)
        self.rec_tree.column("#0", width=0, stretch=tk.NO)
        self.rec_tree.column("ID", width=50, anchor=tk.CENTER)
        self.rec_tree.column("Name", width=200)
        self.rec_tree.column("Genre", width=200)
        self.rec_tree.column("Type", width=80, anchor=tk.CENTER)
        self.rec_tree.column("Episodes", width=80, anchor=tk.CENTER)
        self.rec_tree.column("Rating", width=80, anchor=tk.CENTER)
        
        # Set column headings
        self.rec_tree.heading("ID", text="ID")
        self.rec_tree.heading("Name", text="Name")
        self.rec_tree.heading("Genre", text="Genre")
        self.rec_tree.heading("Type", text="Type")
        self.rec_tree.heading("Episodes", text="Episodes")
        self.rec_tree.heading("Rating", text="Rating")
        
        # Add scrollbar
        rec_scroll = ttk.Scrollbar(rec_results_frame, orient="vertical", command=self.rec_tree.yview)
        rec_scroll.pack(side='right', fill='y')
        self.rec_tree.configure(yscrollcommand=rec_scroll.set)
        
        # Feedback frame
        feedback_frame = ttk.LabelFrame(frame, text="Feedback", padding="10")
        feedback_frame.pack(fill='x', pady=10)
        
        ttk.Label(feedback_frame, text="Select an anime and provide feedback:").pack(anchor='w', padx=5, pady=5)
        
        feedback_btn_frame = ttk.Frame(feedback_frame)
        feedback_btn_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(feedback_btn_frame, text="👍 I like this", command=lambda: self.provide_feedback(True)).pack(side='left', padx=5)
        ttk.Button(feedback_btn_frame, text="👎 Not for me", command=lambda: self.provide_feedback(False)).pack(side='left', padx=5)
        ttk.Button(feedback_btn_frame, text="⭐ Rate", command=self.rate_recommendation).pack(side='left', padx=5)
        
        # Right-click menu for recommendation tree
        self.rec_menu = tk.Menu(self.master, tearoff=0)
        self.rec_menu.add_command(label="I Like This", command=lambda: self.provide_feedback(True))
        self.rec_menu.add_command(label="Not For Me", command=lambda: self.provide_feedback(False))
        self.rec_menu.add_command(label="Rate This Anime", command=self.rate_recommendation)
        self.rec_menu.add_separator()
        self.rec_menu.add_command(label="View Details", command=self.view_recommendation_details)
        
        # Bind right-click event
        self.rec_tree.bind("<Button-3>", self.show_rec_menu)
        self.rec_tree.bind("<Double-1>", lambda e: self.view_recommendation_details())
    
    def setup_profile_tab(self):
        """Set up the user profile interface"""
        frame = ttk.Frame(self.profile_tab, padding="10")
        frame.pack(fill='both', expand=True)
        
        # User preferences
        pref_frame = ttk.LabelFrame(frame, text="Your Preferences", padding="10")
        pref_frame.pack(fill='x', pady=10)
        
        # Favorite genres
        ttk.Label(pref_frame, text="Favorite Genres:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.fav_genres_entry = ttk.Entry(pref_frame, width=40)
        self.fav_genres_entry.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        # Load existing preferences if any
        if 'favorite_genres' in self.recommender.user_profile:
            self.fav_genres_entry.insert(0, self.recommender.user_profile['favorite_genres'])
        
        ttk.Button(pref_frame, text="Browse Genres", command=self.browse_profile_genres).grid(
            row=0, column=2, padx=5, pady=5)
        
        # Preferred type
        ttk.Label(pref_frame, text="Preferred Type:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.pref_type_var = tk.StringVar()
        if 'preferred_type' in self.recommender.user_profile:
            self.pref_type_var.set(self.recommender.user_profile['preferred_type'])
        
        pref_type_options = ["", "TV", "Movie", "OVA", "Special", "ONA", "Music"]
        self.pref_type_combo = ttk.Combobox(pref_frame, textvariable=self.pref_type_var, 
                                             values=pref_type_options, width=10)
        self.pref_type_combo.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        # Episode preference
        ttk.Label(pref_frame, text="Episode Preference:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.ep_pref_var = tk.StringVar()
        if 'episode_preference' in self.recommender.user_profile:
            self.ep_pref_var.set(self.recommender.user_profile['episode_preference'])
        
        ep_options = ["", "Short (1-12)", "Medium (13-24)", "Long (25-50)", "Very Long (51+)"]
        self.ep_pref_combo = ttk.Combobox(pref_frame, textvariable=self.ep_pref_var, values=ep_options, width=15)
        self.ep_pref_combo.grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        # Save preferences button
        save_pref_btn = ttk.Button(pref_frame, text="Save Preferences", command=self.save_preferences)
        save_pref_btn.grid(row=3, column=1, padx=5, pady=10)
        
        # Rating history
        rating_frame = ttk.LabelFrame(frame, text="Your Ratings", padding="10")
        rating_frame.pack(fill='both', expand=True, pady=10)
        
        # Treeview for displaying ratings
        self.rating_tree = ttk.Treeview(rating_frame, columns=("ID", "Name", "Rating", "Date"))
        self.rating_tree.pack(side='left', fill='both', expand=True)
        
        # Configure tree columns
        self.rating_tree.column("#0", width=0, stretch=tk.NO)
        self.rating_tree.column("ID", width=50, anchor=tk.CENTER)
        self.rating_tree.column("Name", width=300)
        self.rating_tree.column("Rating", width=100, anchor=tk.CENTER)
        self.rating_tree.column("Date", width=150, anchor=tk.CENTER)
        
        # Set column headings
        self.rating_tree.heading("ID", text="ID")
        self.rating_tree.heading("Name", text="Anime Name")
        self.rating_tree.heading("Rating", text="Your Rating")
        self.rating_tree.heading("Date", text="Date Rated")
        
        # Add scrollbar
        rating_scroll = ttk.Scrollbar(rating_frame, orient="vertical", command=self.rating_tree.yview)
        rating_scroll.pack(side='right', fill='y')
        self.rating_tree.configure(yscrollcommand=rating_scroll.set)
        
        # Populate rating history
        self.load_rating_history()
        
        # Right-click menu for ratings
        self.rating_menu = tk.Menu(self.master, tearoff=0)
        self.rating_menu.add_command(label="Change Rating", command=self.change_rating)
        self.rating_menu.add_command(label="Remove Rating", command=self.remove_rating)
        
        # Bind right-click event
        self.rating_tree.bind("<Button-3>", self.show_rating_menu)
    
    def setup_stats_tab(self):
        """Set up the analytics interface"""
        frame = ttk.Frame(self.stats_tab, padding="10")
        frame.pack(fill='both', expand=True)
        
        # Create analytics notebook
        analytics_notebook = ttk.Notebook(frame)
        analytics_notebook.pack(fill='both', expand=True)
        
        # Create tabs for different visualizations
        genre_tab = ttk.Frame(analytics_notebook)
        rating_tab = ttk.Frame(analytics_notebook)
        user_tab = ttk.Frame(analytics_notebook)
        
        analytics_notebook.add(genre_tab, text="Genre Analysis")
        analytics_notebook.add(rating_tab, text="Rating Distribution")
        analytics_notebook.add(user_tab, text="Your Activity")
        
        # Set up genre analysis tab
        self.setup_genre_analysis(genre_tab)
        
        # Set up rating distribution tab
        self.setup_rating_analysis(rating_tab)
        
        # Set up user activity tab
        self.setup_user_analysis(user_tab)
    
    def setup_genre_analysis(self, parent):
        """Set up genre analysis visualization"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill='both', expand=True)
        
        # Create figure and canvas for plotting
        self.genre_fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.genre_canvas = FigureCanvasTkAgg(self.genre_fig, frame)
        self.genre_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Plot genre distribution
        self.plot_genre_distribution()
    
    def setup_rating_analysis(self, parent):
        """Set up rating analysis visualization"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill='both', expand=True)
        
        # Create figure and canvas for plotting
        self.rating_fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.rating_canvas = FigureCanvasTkAgg(self.rating_fig, frame)
        self.rating_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Plot rating distribution
        self.plot_rating_distribution()
    
    def setup_user_analysis(self, parent):
        """Set up user activity visualization"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill='both', expand=True)
        
        # Create figure and canvas for plotting
        self.user_fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.user_canvas = FigureCanvasTkAgg(self.user_fig, frame)
        self.user_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Plot user activity
        self.plot_user_activity()
    
    # Event handlers and utility methods
    def browse_genres(self):
        """Show a dialog of available genres to select from"""
        genres = self.recommender.get_genre_statistics()
        genre_window = tk.Toplevel(self.master)
        genre_window.title("Browse Genres")
        genre_window.geometry("400x500")
        
        # Create a listbox with scrollbar
        frame = ttk.Frame(genre_window, padding="10")
        frame.pack(fill='both', expand=True)
        
        ttk.Label(frame, text="Select genres (hold Ctrl for multiple):").pack(anchor='w', pady=(0, 5))
        
        genres_list = ttk.Treeview(frame, columns=("Genre", "Count"), show="headings")
        genres_list.pack(side='left', fill='both', expand=True)
        
        genres_list.column("Genre", width=200)
        genres_list.column("Count", width=100, anchor=tk.CENTER)
        
        genres_list.heading("Genre", text="Genre")
        genres_list.heading("Count", text="Anime Count")
        
        # Add genres to the listbox
        for genre, count in genres:
            genres_list.insert("", "end", values=(genre, count))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=genres_list.yview)
        scrollbar.pack(side='right', fill='y')
        genres_list.configure(yscrollcommand=scrollbar.set)
        
        # Button frame
        btn_frame = ttk.Frame(genre_window, padding="10")
        btn_frame.pack(fill='x')
        
        # Function to add selected genres
        def add_selected_genres():
            selected = [genres_list.item(item)['values'][0] for item in genres_list.selection()]
            if selected:
                current = self.genre_entry.get()
                if current:
                    # Append to existing genres
                    combined = current + ", " + ", ".join(selected)
                    # Remove duplicates while preserving order
                    genres_list = []
                    for g in combined.split(", "):
                        if g.strip() not in genres_list:
                            genres_list.append(g.strip())
                    self.genre_entry.delete(0, tk.END)
                    self.genre_entry.insert(0, ", ".join(genres_list))
                else:
                    # No existing genres
                    self.genre_entry.insert(0, ", ".join(selected))
            genre_window.destroy()
        
        # Select button
        select_btn = ttk.Button(btn_frame, text="Add Selected", command=add_selected_genres)
        select_btn.pack(side='left', padx=5)
        
        # Cancel button
        cancel_btn = ttk.Button(btn_frame, text="Cancel", command=genre_window.destroy)
        cancel_btn.pack(side='right', padx=5)
    
    def browse_profile_genres(self):
        """Browse genres for profile preferences"""
        genres = self.recommender.get_genre_statistics()
        genre_window = tk.Toplevel(self.master)
        genre_window.title("Select Favorite Genres")
        genre_window.geometry("400x500")
        
        # Create a listbox with scrollbar (similar to browse_genres)
        frame = ttk.Frame(genre_window, padding="10")
        frame.pack(fill='both', expand=True)
        
        ttk.Label(frame, text="Select your favorite genres:").pack(anchor='w', pady=(0, 5))
        
        genres_list = ttk.Treeview(frame, columns=("Genre", "Count"), show="headings")
        genres_list.pack(side='left', fill='both', expand=True)
        
        genres_list.column("Genre", width=200)
        genres_list.column("Count", width=100, anchor=tk.CENTER)
        
        genres_list.heading("Genre", text="Genre")
        genres_list.heading("Count", text="Anime Count")
        
        # Add genres to the listbox
        for genre, count in genres:
            genres_list.insert("", "end", values=(genre, count))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=genres_list.yview)
        scrollbar.pack(side='right', fill='y')
        genres_list.configure(yscrollcommand=scrollbar.set)
        
        # Button frame
        btn_frame = ttk.Frame(genre_window, padding="10")
        btn_frame.pack(fill='x')
        
        # Function to add selected genres to profile
        def add_selected_genres():
            selected = [genres_list.item(item)['values'][0] for item in genres_list.selection()]
            if selected:
                self.fav_genres_entry.delete(0, tk.END)
                self.fav_genres_entry.insert(0, ", ".join(selected))
            genre_window.destroy()
        
        # Select button
        select_btn = ttk.Button(btn_frame, text="Select Genres", command=add_selected_genres)
        select_btn.pack(side='left', padx=5)
        
        # Cancel button
        cancel_btn = ttk.Button(btn_frame, text="Cancel", command=genre_window.destroy)
        cancel_btn.pack(side='right', padx=5)
    
    def browse_anime(self):
        """Browse anime to select one as a base for recommendations"""
        browse_window = tk.Toplevel(self.master)
        browse_window.title("Browse Anime")
        browse_window.geometry("800x500")
        
        # Create a frame for search controls
        search_frame = ttk.Frame(browse_window, padding="10")
        search_frame.pack(fill='x')
        
        ttk.Label(search_frame, text="Search:").pack(side='left', padx=5)
        search_entry = ttk.Entry(search_frame, width=30)
        search_entry.pack(side='left', padx=5)
        
        # Create a frame for the anime list
        list_frame = ttk.Frame(browse_window, padding="10")
        list_frame.pack(fill='both', expand=True)
        
        # Create a treeview for displaying anime
        anime_list = ttk.Treeview(list_frame, columns=("ID", "Name", "Genre", "Rating"), show="headings")
        anime_list.pack(side='left', fill='both', expand=True)
        
        anime_list.column("ID", width=50, anchor=tk.CENTER)
        anime_list.column("Name", width=300)
        anime_list.column("Genre", width=300)
        anime_list.column("Rating", width=80, anchor=tk.CENTER)
        
        anime_list.heading("ID", text="ID")
        anime_list.heading("Name", text="Name")
        anime_list.heading("Genre", text="Genre")
        anime_list.heading("Rating", text="Rating")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=anime_list.yview)
        scrollbar.pack(side='right', fill='y')
        anime_list.configure(yscrollcommand=scrollbar.set)
        
        # Function to search anime
        def search_anime():
            query = search_entry.get().lower()
            anime_list.delete(*anime_list.get_children())
            
            # Get top 100 anime by rating if no search term
            if not query:
                df_sorted = self.recommender.df.sort_values('rating', ascending=False).head(100)
            else:
                df_sorted = self.recommender.df[self.recommender.df['name'].str.contains(query, case=False, na=False)]
            
            # Populate the treeview
            for _, row in df_sorted.iterrows():
                anime_list.insert("", "end", values=(
                    row['anime_id'],
                    row['name'],
                    row['genre'],
                    row['rating']
                ))
        
        # Search button
        search_btn = ttk.Button(search_frame, text="Search", command=search_anime)
        search_btn.pack(side='left', padx=5)
        
        # Button frame
        btn_frame = ttk.Frame(browse_window, padding="10")
        btn_frame.pack(fill='x')
        
        # Function to select an anime
        def select_anime():
            selected = anime_list.focus()
            if selected:
                values = anime_list.item(selected)['values']
                anime_id = values[0]
                anime_name = values[1]
                
                # Set the selected anime in the entry field
                self.base_anime_entry.delete(0, tk.END)
                self.base_anime_entry.insert(0, f"{anime_id} - {anime_name}")
                
                browse_window.destroy()
        
        # Select button
        select_btn = ttk.Button(btn_frame, text="Select", command=select_anime)
        select_btn.pack(side='left', padx=5)
        
        # Cancel button
        cancel_btn = ttk.Button(btn_frame, text="Cancel", command=browse_window.destroy)
        cancel_btn.pack(side='right', padx=5)
        
        # Initial load of top anime
        search_anime()
    
    def perform_search(self):
        """Execute search based on criteria"""
        # Clear existing results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Gather search criteria
        criteria = {
            'name': self.name_entry.get(),
            'genre': self.genre_entry.get(),
            'type': self.type_var.get(),
            'min_rating': self.min_rating_var.get(),
            'max_rating': self.max_rating_var.get(),
            'min_episodes': self.min_episodes_var.get(),
            'max_episodes': self.max_episodes_var.get(),
            'sort_by': self.sort_var.get() or 'rating'
        }
        
        # Perform search
        results = self.recommender.search_anime(criteria)
        
        # Display results
        if len(results) > 0:
            for _, row in results.iterrows():
                self.results_tree.insert("", "end", values=(
                    row['anime_id'],
                    row['name'],
                    row['genre'],
                    row['type'],
                    row['episodes'],
                    round(row['rating'], 2) if pd.notna(row['rating']) else "N/A"
                ))
            messagebox.showinfo("Search Results", f"Found {len(results)} anime matching your criteria.")
        else:
            messagebox.showinfo("Search Results", "No anime found matching your criteria.")
    
    def clear_search(self):
        """Clear all search fields"""
        self.name_entry.delete(0, tk.END)
        self.genre_entry.delete(0, tk.END)
        self.type_var.set("")
        self.min_rating_var.set("")
        self.max_rating_var.set("")
        self.min_episodes_var.set("")
        self.max_episodes_var.set("")
        self.sort_var.set("")
        
        # Clear results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
    
    def show_tree_menu(self, event):
        """Show context menu for search results tree"""
        # Select row under mouse
        iid = self.results_tree.identify_row(event.y)
        if iid:
            self.results_tree.selection_set(iid)
            self.tree_menu.post(event.x_root, event.y_root)
    
    def show_rec_menu(self, event):
        """Show context menu for recommendations tree"""
        iid = self.rec_tree.identify_row(event.y)
        if iid:
            self.rec_tree.selection_set(iid)
            self.rec_menu.post(event.x_root, event.y_root)
    
    def show_rating_menu(self, event):
        """Show context menu for ratings tree"""
        iid = self.rating_tree.identify_row(event.y)
        if iid:
            self.rating_tree.selection_set(iid)
            self.rating_menu.post(event.x_root, event.y_root)
    
    def recommend_similar(self):
        """Recommend anime similar to selected anime"""
        selected = self.results_tree.focus()
        if not selected:
            messagebox.showinfo("Select Anime", "Please select an anime first.")
            return
        
        # Get anime ID from selected row
        anime_id = self.results_tree.item(selected)['values'][0]
        anime_name = self.results_tree.item(selected)['values'][1]
        
        # Set as base anime in recommendations tab and switch to that tab
        self.base_anime_entry.delete(0, tk.END)
        self.base_anime_entry.insert(0, f"{anime_id} - {anime_name}")
        self.rec_type_var.set("content")
        self.notebook.select(self.recommendations_tab)
        
        # Trigger recommendations
        self.get_recommendations()
    
    def rate_anime(self):
        """Rate the selected anime from search results"""
        selected = self.results_tree.focus()
        if not selected:
            messagebox.showinfo("Select Anime", "Please select an anime first.")
            return
        
        # Get anime ID and name
        values = self.results_tree.item(selected)['values']
        anime_id = values[0]
        anime_name = values[1]
        
        # Ask for rating
        rating = simpledialog.askfloat(
            "Rate Anime", 
            f"Rate '{anime_name}' (1-10):", 
            minvalue=1, 
            maxvalue=10
        )
        
        if rating is not None:
            # Record the rating
            if self.recommender.record_rating(anime_id, rating):
                messagebox.showinfo("Rating Recorded", f"Your rating of {rating} for '{anime_name}' has been recorded.")
                
                # Refresh rating history if we're on that tab
                if self.notebook.index(self.notebook.select()) == 2:  # Profile tab
                    self.load_rating_history()
            else:
                messagebox.showerror("Error", "Could not record rating.")
    
    def rate_recommendation(self):
        """Rate a recommended anime"""
        selected = self.rec_tree.focus()
        if not selected:
            messagebox.showinfo("Select Anime", "Please select an anime first.")
            return
        
        # Get anime ID and name
        values = self.rec_tree.item(selected)['values']
        anime_id = values[0]
        anime_name = values[1]
        
        # Ask for rating
        rating = simpledialog.askfloat(
            "Rate Anime", 
            f"Rate '{anime_name}' (1-10):", 
            minvalue=1, 
            maxvalue=10
        )
        
        if rating is not None:
            # Record the rating
            if self.recommender.record_rating(anime_id, rating):
                messagebox.showinfo("Rating Recorded", f"Your rating of {rating} for '{anime_name}' has been recorded.")
                
                # Record feedback based on rating
                liked = rating >= 7.0  # Consider 7+ as "liked"
                self.recommender.record_feedback(anime_id, liked, f"Rated {rating}/10")
                
                # Refresh rating history if we're on that tab
                if self.notebook.index(self.notebook.select()) == 2:  # Profile tab
                    self.load_rating_history()
            else:
                messagebox.showerror("Error", "Could not record rating.")
    
    def provide_feedback(self, liked):
        """Provide feedback on a recommendation"""
        selected = self.rec_tree.focus()
        if not selected:
            messagebox.showinfo("Select Anime", "Please select an anime first.")
            return
        
        # Get anime ID and name
        values = self.rec_tree.item(selected)['values']
        anime_id = values[0]
        anime_name = values[1]
        
        # Option to add comment
        comment = ""
        if liked:
            comment = simpledialog.askstring(
                "Feedback", 
                f"What did you like about '{anime_name}'? (optional)"
            ) or ""
        else:
            comment = simpledialog.askstring(
                "Feedback", 
                f"What didn't you like about '{anime_name}'? (optional)"
            ) or ""
        
        # Record the feedback
        if self.recommender.record_feedback(anime_id, liked, comment):
            if liked:
                messagebox.showinfo("Feedback Recorded", 
                                   f"You liked '{anime_name}'. This will help improve your recommendations.")
            else:
                messagebox.showinfo("Feedback Recorded", 
                                   f"You didn't like '{anime_name}'. This will help improve your recommendations.")
        else:
            messagebox.showerror("Error", "Could not record feedback.")
    
    def view_anime_details(self):
        """View details of selected anime from search results"""
        selected = self.results_tree.focus()
        if not selected:
            messagebox.showinfo("Select Anime", "Please select an anime first.")
            return
        
        # Get anime details
        values = self.results_tree.item(selected)['values']
        anime_id = values[0]
        anime_name = values[1]
        anime_genre = values[2]
        anime_type = values[3]
        anime_episodes = values[4]
        anime_rating = values[5]
        
        # Display details in a dialog
        details = f"ID: {anime_id}\n"
        details += f"Name: {anime_name}\n"
        details += f"Genre: {anime_genre}\n"
        details += f"Type: {anime_type}\n"
        details += f"Episodes: {anime_episodes}\n"
        details += f"Rating: {anime_rating}\n"
        
        # Create a custom dialog
        details_window = tk.Toplevel(self.master)
        details_window.title(f"Anime Details: {anime_name}")
        details_window.geometry("500x300")
        
        # Add details text
        ttk.Label(details_window, text="Anime Details", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        details_frame = ttk.Frame(details_window, padding="20")
        details_frame.pack(fill='both', expand=True)
        
        # Add each detail as a separate label for better formatting
        ttk.Label(details_frame, text=f"ID: {anime_id}", font=("Helvetica", 11)).pack(anchor='w', pady=2)
        ttk.Label(details_frame, text=f"Name: {anime_name}", font=("Helvetica", 11, "bold")).pack(anchor='w', pady=2)
        ttk.Label(details_frame, text=f"Genre: {anime_genre}", font=("Helvetica", 11)).pack(anchor='w', pady=2)
        ttk.Label(details_frame, text=f"Type: {anime_type}", font=("Helvetica", 11)).pack(anchor='w', pady=2)
        ttk.Label(details_frame, text=f"Episodes: {anime_episodes}", font=("Helvetica", 11)).pack(anchor='w', pady=2)
        ttk.Label(details_frame, text=f"Rating: {anime_rating}", font=("Helvetica", 11)).pack(anchor='w', pady=2)
        
        # Action buttons
        btn_frame = ttk.Frame(details_window, padding="10")
        btn_frame.pack(fill='x')
        
        ttk.Button(btn_frame, text="Find Similar", 
                   command=lambda: [details_window.destroy(), self.recommend_similar()]).pack(side='left', padx=5)
        
        ttk.Button(btn_frame, text="Rate", 
                   command=lambda: [details_window.destroy(), self.rate_anime()]).pack(side='left', padx=5)
        
        ttk.Button(btn_frame, text="Close", 
                   command=details_window.destroy).pack(side='right', padx=5)
    
    def view_recommendation_details(self):
        """View details of selected recommendation"""
        selected = self.rec_tree.focus()
        if not selected:
            messagebox.showinfo("Select Anime", "Please select an anime first.")
            return
        
        # Get anime details (similar to view_anime_details)
        values = self.rec_tree.item(selected)['values']
        anime_id = values[0]
        anime_name = values[1]
        anime_genre = values[2]
        anime_type = values[3]
        anime_episodes = values[4]
        anime_rating = values[5]
        
        # Display details in a dialog (largely the same as view_anime_details)
        details_window = tk.Toplevel(self.master)
        details_window.title(f"Anime Details: {anime_name}")
        details_window.geometry("500x350")
        
        # Add details text
        ttk.Label(details_window, text="Recommended Anime", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        details_frame = ttk.Frame(details_window, padding="20")
        details_frame.pack(fill='both', expand=True)
        
        # Add each detail as a separate label
        ttk.Label(details_frame, text=f"ID: {anime_id}", font=("Helvetica", 11)).pack(anchor='w', pady=2)
        ttk.Label(details_frame, text=f"Name: {anime_name}", font=("Helvetica", 11, "bold")).pack(anchor='w', pady=2)
        ttk.Label(details_frame, text=f"Genre: {anime_genre}", font=("Helvetica", 11)).pack(anchor='w', pady=2)
        ttk.Label(details_frame, text=f"Type: {anime_type}", font=("Helvetica", 11)).pack(anchor='w', pady=2)
        ttk.Label(details_frame, text=f"Episodes: {anime_episodes}", font=("Helvetica", 11)).pack(anchor='w', pady=2)
        ttk.Label(details_frame, text=f"Rating: {anime_rating}", font=("Helvetica", 11)).pack(anchor='w', pady=2)
        
        # Feedback section
        feedback_frame = ttk.LabelFrame(details_window, text="Your Feedback", padding="10")
        feedback_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Button(feedback_frame, text="👍 I like this", 
                   command=lambda: [details_window.destroy(), self.provide_feedback(True)]).pack(side='left', padx=5)
        
        ttk.Button(feedback_frame, text="👎 Not for me", 
                   command=lambda: [details_window.destroy(), self.provide_feedback(False)]).pack(side='left', padx=5)
        
        ttk.Button(feedback_frame, text="⭐ Rate", 
                   command=lambda: [details_window.destroy(), self.rate_recommendation()]).pack(side='left', padx=5)
        
        ttk.Button(feedback_frame, text="Close", 
                   command=details_window.destroy).pack(side='right', padx=5)
    
    def get_recommendations(self):
        """Get anime recommendations based on selected criteria"""
        # Clear existing recommendations
        for item in self.rec_tree.get_children():
            self.rec_tree.delete(item)
        
        # Get recommendation type
        rec_type = self.rec_type_var.get()
        
        # Get base anime for content-based recommendations
        anime_id = None
        if rec_type in ["content", "hybrid"]:
            base_anime = self.base_anime_entry.get()
            if base_anime:
                # Extract anime ID from the entry (format: "ID - Name")
                try:
                    anime_id = int(base_anime.split(" - ")[0])
                except (ValueError, IndexError):
                    # Try to see if it's just an ID
                    try:
                        anime_id = int(base_anime)
                    except ValueError:
                        messagebox.showerror("Invalid Anime", 
                                           "Please select a valid anime using the Browse button.")
                        return
            elif rec_type == "content":
                messagebox.showerror("Base Anime Required", 
                                   "Please select a base anime for content-based recommendations.")
                return
        
        # Get number of recommendations
        try:
            num_recs = int(self.num_recs_var.get())
        except ValueError:
            num_recs = 10  # Default
        
        # Get recommendations based on type
        if rec_type == "content" and anime_id:
            recommendations = self.recommender.get_content_recommendations(anime_id, top_n=num_recs)
        elif rec_type == "collaborative":
            recommendations = self.recommender.get_collaborative_recommendations(top_n=num_recs)
        else:  # hybrid or fallback
            recommendations = self.recommender.get_hybrid_recommendations(anime_id, top_n=num_recs)
        
        # Display recommendations
        if len(recommendations) > 0:
            for _, row in recommendations.iterrows():
                self.rec_tree.insert("", "end", values=(
                    row['anime_id'],
                    row['name'],
                    row['genre'],
                    row['type'],
                    row['episodes'],
                    round(row['rating'], 2) if pd.notna(row['rating']) else "N/A"
                ))
            messagebox.showinfo("Recommendations", f"Found {len(recommendations)} anime recommendations.")
        else:
            messagebox.showinfo("Recommendations", "No recommendations found. Try different criteria.")
    
    def save_preferences(self):
        """Save user preferences to profile"""
        preferences = {
            'favorite_genres': self.fav_genres_entry.get(),
            'preferred_type': self.pref_type_var.get(),
            'episode_preference': self.ep_pref_var.get()
        }
        
        if self.recommender.update_user_profile(preferences):
            messagebox.showinfo("Preferences Saved", "Your preferences have been saved and will be used for recommendations.")
        else:
            messagebox.showerror("Error", "Could not save preferences.")
    
    def load_rating_history(self):
        """Load and display user rating history"""
        # Clear existing ratings
        for item in self.rating_tree.get_children():
            self.rating_tree.delete(item)
        
        # Add ratings to the tree
        for anime_id, rating in self.recommender.user_ratings.items():
            try:
                # Get anime details
                anime_id_int = int(anime_id)
                anime_row = self.recommender.df[self.recommender.df['anime_id'] == anime_id_int]
                
                if not anime_row.empty:
                    anime_name = anime_row.iloc[0]['name']
                    
                    # Add to tree (we don't have actual date info, so using placeholder)
                    self.rating_tree.insert("", "end", values=(
                        anime_id,
                        anime_name,
                        f"{rating:.1f}/10",
                        "N/A"  # We don't store rating dates in this simple version
                    ))
            except (ValueError, IndexError):
                continue
    
    def change_rating(self):
        """Change a previously recorded rating"""
        selected = self.rating_tree.focus()
        if not selected:
            messagebox.showinfo("Select Rating", "Please select a rating to change.")
            return
        
        # Get anime ID and current rating
        values = self.rating_tree.item(selected)['values']
        anime_id = values[0]
        anime_name = values[1]
        current_rating = float(values[2].split('/')[0])
        
        # Ask for new rating
        new_rating = simpledialog.askfloat(
            "Change Rating", 
            f"Current rating for '{anime_name}': {current_rating}\nNew rating (1-10):", 
            minvalue=1, 
            maxvalue=10
        )
        
        if new_rating is not None:
            # Update the rating
            if self.recommender.record_rating(anime_id, new_rating):
                messagebox.showinfo("Rating Updated", f"Your rating for '{anime_name}' has been updated to {new_rating}.")
                self.load_rating_history()  # Refresh the list
            else:
                messagebox.showerror("Error", "Could not update rating.")
    
    def remove_rating(self):
        """Remove a rating from history"""
        selected = self.rating_tree.focus()
        if not selected:
            messagebox.showinfo("Select Rating", "Please select a rating to remove.")
            return
        
        # Get anime ID
        anime_id = self.rating_tree.item(selected)['values'][0]
        anime_name = self.rating_tree.item(selected)['values'][1]
        
        # Confirm removal
        if messagebox.askyesno("Confirm Removal", f"Remove your rating for '{anime_name}'?"):
            # Remove the rating
            if str(anime_id) in self.recommender.user_ratings:
                del self.recommender.user_ratings[str(anime_id)]
                self.recommender.save_user_data()
                messagebox.showinfo("Rating Removed", f"Your rating for '{anime_name}' has been removed.")
                self.load_rating_history()  # Refresh the list
            else:
                messagebox.showerror("Error", "Could not find rating to remove.")
    
    def plot_genre_distribution(self):
        """Plot the distribution of anime genres"""
        # Get genre data
        genre_data = self.recommender.get_genre_statistics()
        
        # Take top 15 genres for better visualization
        top_genres = genre_data[:15]
        genres = [g[0] for g in top_genres]
        counts = [g[1] for g in top_genres]
        
        # Clear previous plot
        self.genre_fig.clear()
        
        # Create new plot
        ax = self.genre_fig.add_subplot(111)
        bars = ax.barh(genres, counts, color='royalblue')
        
        # Add count labels to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 10, bar.get_y() + bar.get_height()/2, 
                   f'{int(width)}', ha='left', va='center')
        
        # Set labels and title
        ax.set_title('Top 15 Anime Genres', fontsize=14)
        ax.set_xlabel('Number of Anime', fontsize=12)
        
        # Adjust layout
        self.genre_fig.tight_layout()
        self.genre_canvas.draw()
    
    def plot_rating_distribution(self):
        """Plot the distribution of anime ratings"""
        # Get rating data
        ratings = self.recommender.df['rating'].dropna()
        
        # Clear previous plot
        self.rating_fig.clear()
        
        # Create new plot
        ax = self.rating_fig.add_subplot(111)
        
        # Create histogram
        n, bins, patches = ax.hist(ratings, bins=20, color='teal', alpha=0.7)
        
        # Add a line showing the mean rating
        mean_rating = ratings.mean()
        ax.axvline(mean_rating, color='red', linestyle='dashed', linewidth=1)
        ax.text(mean_rating + 0.1, max(n) * 0.9, f'Mean: {mean_rating:.2f}', 
               color='red', fontweight='bold')
        
        # Set labels and title
        ax.set_title('Anime Rating Distribution', fontsize=14)
        ax.set_xlabel('Rating', fontsize=12)
        ax.set_ylabel('Number of Anime', fontsize=12)
        
        # Adjust layout
        self.rating_fig.tight_layout()
        self.rating_canvas.draw()
    
    def plot_user_activity(self):
        """Plot user activity statistics"""
        # Clear previous plot
        self.user_fig.clear()
        
        # Create subplots for different metrics
        if self.recommender.user_ratings:
            # User ratings distribution
            ratings = list(self.recommender.user_ratings.values())
            
            ax1 = self.user_fig.add_subplot(121)
            ax1.hist(ratings, bins=10, range=(1, 10), color='purple', alpha=0.7)
            ax1.set_title('Your Rating Distribution', fontsize=12)
            ax1.set_xlabel('Rating', fontsize=10)
            ax1.set_ylabel('Count', fontsize=10)
            
            # User genres preference (from rated anime)
            genre_counts = {}
            for anime_id, rating in self.recommender.user_ratings.items():
                try:
                    anime_id_int = int(anime_id)
                    anime_row = self.recommender.df[self.recommender.df['anime_id'] == anime_id_int]
                    
                    if not anime_row.empty:
                        genres = anime_row.iloc[0]['genre']
                        if isinstance(genres, str):
                            for genre in genres.split(','):
                                genre = genre.strip()
                                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                except (ValueError, IndexError):
                    continue
            
            # Take top 8 genres for better visualization
            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            
            if top_genres:
                ax2 = self.user_fig.add_subplot(122)
                genres = [g[0] for g in top_genres]
                counts = [g[1] for g in top_genres]
                
                ax2.pie(counts, labels=genres, autopct='%1.1f%%', 
                       startangle=90, shadow=True)
                ax2.set_title('Your Genre Preferences', fontsize=12)
                
                # Equal aspect ratio ensures that pie is drawn as a circle
                ax2.axis('equal')
        else:
            # No user data yet
            ax = self.user_fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Rate some anime to see your activity statistics', 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=14)
            ax.axis('off')
        
        # Adjust layout
        self.user_fig.tight_layout()
        self.user_canvas.draw()

# Main entry point
def main():
    root = tk.Tk()
    app = AnimeRecommenderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()