import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import logging
from anime_data import AnimeData, UserData
from recommendation_engine import RecommendationEngine
from gui_components import LoadingDialog, SearchTab, RecommendationsTab, ProfileTab, AnalyticsTab

class AnimeRecommenderApp:
    """Main application class for the Anime Recommendation System"""
    def __init__(self, root):
        """Initialize the application with the root window"""
        self.root = root
        self.root.title("Advanced Anime Recommendation System")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        
        # Set up data loading
        self.setup_data()
        
        # Create main interface after data is loaded
        self.create_ui()
        
    def setup_data(self):
        """Load data and initialize recommendation engine"""
        # Show loading dialog
        self.loading = LoadingDialog(self.root, message="Loading anime data...")
        self.root.update_idletasks()
        
        # Load data in a background thread
        self.data_loaded = False
        
        def load_data_thread():
            try:
                # Create data objects
                self.anime_data = AnimeData()
                self.user_data = UserData()
                
                # Create recommendation engine
                self.recommendation_engine = RecommendationEngine(self.anime_data, self.user_data)
                
                # Flag data as loaded
                self.data_loaded = True
                
                # Update UI in main thread
                self.root.after(100, self.finish_loading)
            except Exception as e:
                # Handle errors
                logging.error(f"Error loading data: {e}")
                self.root.after(100, lambda: self.handle_loading_error(str(e)))
        
        # Start loading thread
        self.load_thread = threading.Thread(target=load_data_thread)
        self.load_thread.daemon = True
        self.load_thread.start()
    
    def finish_loading(self):
        """Close loading dialog and continue UI setup"""
        self.loading.close()
        
        # Initialize models in background
        self.initialize_models()
        
        # Create the UI
        self.create_ui()
    
    def handle_loading_error(self, error_message):
        """Handle errors during data loading"""
        self.loading.close()
        messagebox.showerror("Loading Error", 
                           f"Error loading anime data: {error_message}\n\nPlease check that anime.csv exists in the application directory.")
        self.root.quit()
    
    def initialize_models(self):
        """Initialize recommendation models in background"""
        def init_thread():
            # Train content-based model if needed
            if 'content_model' not in self.recommendation_engine.models:
                self.recommendation_engine.train_content_based_model()
            
            # Train collaborative model if possible
            if len(self.user_data.user_ratings) >= 5 and 'collaborative_model' not in self.recommendation_engine.models:
                self.recommendation_engine.train_collaborative_model()
        
        # Start initialization thread
        init_thread = threading.Thread(target=init_thread)
        init_thread.daemon = True
        init_thread.start()
    
    def create_ui(self):
        """Create the main user interface"""
        # Create tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.search_tab = SearchTab(self.notebook, self.anime_data, self.recommendation_engine)
        self.recommendations_tab = RecommendationsTab(self.notebook, self.anime_data, self.recommendation_engine)
        self.profile_tab = ProfileTab(self.notebook, self.anime_data, self.user_data)
        self.analytics_tab = AnalyticsTab(self.notebook, self.anime_data, self.user_data)
        
        # Add tabs to notebook
        self.notebook.add(self.search_tab, text="Search Anime")
        self.notebook.add(self.recommendations_tab, text="Get Recommendations")
        self.notebook.add(self.profile_tab, text="User Profile")
        self.notebook.add(self.analytics_tab, text="Analytics")
        
        # Set up event bindings for communication between tabs
        self.setup_event_bindings()
        
        # Add status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_event_bindings(self):
        """Set up event bindings for communication between tabs"""
        # When search tab wants to show recommendations
        self.search_tab.bind("<<ShowRecommendations>>", self.handle_show_recommendations)
        
        # When ratings are changed in any tab
        self.search_tab.bind("<<RatingsChanged>>", self.handle_ratings_changed)
        self.recommendations_tab.bind("<<RatingsChanged>>", self.handle_ratings_changed)
        self.profile_tab.bind("<<RatingsChanged>>", self.handle_ratings_changed)
    
    def handle_show_recommendations(self, event):
        """Handle request to show recommendations for a specific anime"""
        # Get anime ID and name from event data
        data = event.widget.tk.call("set", "::tk::Priv(x-set-var)")
        if data:
            anime_id, anime_name = data.split("|", 1)
            
            # Switch to recommendations tab
            self.notebook.select(1)  # Index 1 is the recommendations tab
            
            # Set base anime in recommendations tab
            self.recommendations_tab.set_base_anime(int(anime_id), anime_name)
    
    def handle_ratings_changed(self, event):
        """Handle when user ratings have changed"""
        # Refresh rating history in profile tab
        self.profile_tab.load_rating_history()
        
        # Refresh user activity plot
        self.analytics_tab.plot_user_activity()
        
        # Update status
        self.status_bar.config(text=f"User ratings updated - {len(self.user_data.user_ratings)} total ratings")
        
        # Re-train collaborative model if needed
        if len(self.user_data.user_ratings) >= 5:
            def train_thread():
                self.recommendation_engine.train_collaborative_model()
            
            # Start training in background
            thread = threading.Thread(target=train_thread)
            thread.daemon = True
            thread.start()

def main():
    """Main entry point for the application"""
    # Create root window
    root = tk.Tk()
    
    # Set theme (if available)
    try:
        style = ttk.Style()
        style.theme_use('clam')  # Try to use a modern theme
    except:
        pass  # Fallback to default theme
    
    # Create application
    app = AnimeRecommenderApp(root)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()