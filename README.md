# Advanced Anime Recommendation System

A sophisticated machine learning-based recommendation engine that delivers personalized anime suggestions through multiple recommendation paradigms.

## Features

- **Hybrid Recommendation Engine**
  - Content-based filtering: Suggests anime similar to ones you like based on genres, type, and other features
  - Collaborative filtering: Recommends anime based on your ratings and user preference patterns
  - Dimension reduction for improved algorithm performance

- **Machine Learning Components**
  - Feature extraction from anime metadata (genres, types, episodes, etc.)
  - Similarity calculation using cosine similarity metrics
  - K-means clustering for grouping similar anime
  - Nearest neighbors model for collaborative filtering

- **User-Friendly GUI**
  - Modern tabbed interface for easy navigation
  - Advanced search with multiple filtering options
  - Detailed anime information displays
  - Recommendation system with multiple filtering options
  - User profile management
  - Visual analytics and statistics

- **Learning and Feedback Loop**
  - Rating system to record your preferences
  - Feedback mechanism (like/dislike) to improve recommendations
  - Preference tracking to adjust recommendation algorithms
  - History tracking to avoid repetitive suggestions

- **Data Visualization and Analytics**
  - Genre distribution charts
  - Rating distribution analysis
  - User activity and preference tracking
  - Visual representation of recommendation patterns

## Installation

1. **Prerequisites:** Make sure you have Python 3.6+ installed on your system.

2. **Install required packages:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

3. **Install Tkinter (if not already installed):**
   - Windows: Should be included with Python
   - macOS: `brew install python-tk`
   - Ubuntu/Debian: `sudo apt-get install python3-tk`
   - Fedora/RHEL: `sudo dnf install python3-tkinter`

4. **Download files:**
   - Place all Python files in the same directory
   - Ensure `anime.csv` is in the same directory as the scripts

5. **Create models directory:**
   ```bash
   mkdir models
   ```

## Usage

Run the application by executing:
```bash
python main.py
```

### Search Tab
- Find anime by name, genre, type, rating, or episode count
- View detailed information about specific anime
- Rate anime directly from search results
- Find similar anime with one click

### Recommendations Tab
- Get personalized recommendations using different algorithms
- Choose between content-based, collaborative, or hybrid recommendations
- Provide feedback on recommendations to improve future suggestions
- Rate recommended anime

### Profile Tab
- Set your genre and viewing preferences
- View and manage your rating history
- Change or remove previous ratings

### Analytics Tab
- Explore genre distribution in the anime database
- View rating distribution across all anime
- Analyze your own rating patterns and preferences

## Project Structure

- `main.py` - Main application entry point
- `anime_data.py` - Data loading and management
- `recommendation_engine.py` - Machine learning recommendation algorithms
- `gui_components.py` - UI components and visualization
- `anime.csv` - Dataset containing anime information
- `models/` - Directory where trained models are saved
- `user_data.json` - User ratings and preferences (created on first run)

## How It Works

1. **Content-Based Filtering:**
   - Analyzes anime features (genres, type, episodes, etc.)
   - Calculates similarity between different anime using cosine similarity
   - Recommends anime similar to ones you've liked or selected

2. **Collaborative Filtering:**
   - Uses your rating history to identify patterns
   - Creates a user profile based on your preferences
   - Finds anime that match your taste profile

3. **Hybrid Recommendations:**
   - Combines results from both methods for better suggestions
   - Weights results based on confidence scores
   - Learns from your feedback to improve over time

4. **Machine Learning Components:**
   - Feature extraction and normalization
   - Dimensionality reduction with PCA
   - Similarity calculation with optimized algorithms
   - Clustering for better recommendation grouping

## Data Sources

The application uses the `anime.csv` dataset, which contains information about thousands of anime, including:
- Title
- Genres
- Type (TV, Movie, OVA, etc.)
- Number of episodes
- Ratings
- Popularity statistics

## Acknowledgments

- Dataset source: Kaggle Anime Recommendations Database
- Special thanks to the scikit-learn and pandas development teams for their excellent libraries