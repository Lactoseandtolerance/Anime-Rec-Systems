import os
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='anime_recommender.log')

class AnimeData:
    """Class for loading and processing anime data"""
    def __init__(self, csv_path="anime.csv"):
        """Initialize with the dataset path"""
        self.data_path = csv_path
        self.df = None
        self.all_genres = set()
        self.load_data()
        
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
            return True
        except Exception as e:
            logging.error(f"Error loading data: {e}")
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


class UserData:
    """Class for managing user preferences and ratings"""
    def __init__(self):
        """Initialize user data structures"""
        self.user_profile = {}
        self.user_ratings = {}
        self.feedback_history = []
        self.recommendation_history = []
        self.load_user_data()
    
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
                return True
            except Exception as e:
                logging.error(f"Error loading user data: {e}")
                return False
        return False
    
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
            return True
        except Exception as e:
            logging.error(f"Error saving user data: {e}")
            return False
    
    def record_rating(self, anime_id, rating):
        """Record user rating for an anime"""
        try:
            anime_id = int(anime_id)
            rating = float(rating)
            if 1 <= rating <= 10:
                self.user_ratings[str(anime_id)] = rating
                logging.info(f"Recorded rating {rating} for anime {anime_id}")
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
            if liked:
                # For simplicity, just record a positive rating
                if str(anime_id) not in self.user_ratings:
                    self.user_ratings[str(anime_id)] = 8.0
            else:
                # For simplicity, just record a low rating
                if str(anime_id) not in self.user_ratings:
                    self.user_ratings[str(anime_id)] = 3.0
            
            self.save_user_data()
            return True
        except Exception as e:
            logging.error(f"Error recording feedback: {e}")
            return False
    
    def update_profile(self, preferences):
        """Update user profile with genre and type preferences"""
        try:
            self.user_profile.update(preferences)
            logging.info(f"Updated user profile: {preferences}")
            self.save_user_data()
            return True
        except Exception as e:
            logging.error(f"Error updating user profile: {e}")
            return False