import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import logging

class LoadingDialog:
    """Simple loading dialog to show during long operations"""
    def __init__(self, parent, title="Loading", message="Please wait..."):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("300x100")
        self.top.resizable(False, False)
        self.top.transient(parent)
        self.top.grab_set()
        
        # Center the window
        self.top.update_idletasks()
        width = self.top.winfo_width()
        height = self.top.winfo_height()
        x = (self.top.winfo_screenwidth() // 2) - (width // 2)
        y = (self.top.winfo_screenheight() // 2) - (height // 2)
        self.top.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
        # Add message
        ttk.Label(self.top, text=message, font=("Helvetica", 12)).pack(pady=20)
        
        # Make it modal
        self.top.focus_set()
    
    def close(self):
        """Close the dialog"""
        self.top.destroy()

class SearchTab(ttk.Frame):
    """Tab for searching anime"""
    def __init__(self, parent, anime_data, recommendation_engine):
        super().__init__(parent, padding="10")
        self.anime_data = anime_data
        self.recommendation_engine = recommendation_engine
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the search interface"""
        # Search criteria
        criteria_frame = ttk.LabelFrame(self, text="Search Criteria", padding="10")
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
        results_frame = ttk.LabelFrame(self, text="Search Results", padding="10")
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
        self.tree_menu = tk.Menu(self, tearoff=0)
        self.tree_menu.add_command(label="Recommend Similar Anime", command=self.recommend_similar)
        self.tree_menu.add_command(label="Rate This Anime", command=self.rate_anime)
        self.tree_menu.add_separator()
        self.tree_menu.add_command(label="View Details", command=self.view_anime_details)
        
        # Bind right-click event
        self.results_tree.bind("<Button-3>", self.show_tree_menu)
        self.results_tree.bind("<Double-1>", lambda e: self.view_anime_details())
    
    def browse_genres(self):
        """Show a dialog of available genres to select from"""
        genres = self.anime_data.get_genre_statistics()
        genre_window = tk.Toplevel(self)
        genre_window.title("Browse Genres")
        genre_window.geometry("400x500")
        
        # Create a listbox with scrollbar
        frame = ttk.Frame(genre_window, padding="10")
        frame.pack(fill='both', expand=True)
        
        ttk.Label(frame, text="Select genres (hold Ctrl for multiple):").pack(anchor='w', pady=(0, 5))
        
        # Use instance variable to avoid scope issues
        self.temp_genres_list = ttk.Treeview(frame, columns=("Genre", "Count"), show="headings")
        self.temp_genres_list.pack(side='left', fill='both', expand=True)
        
        self.temp_genres_list.column("Genre", width=200)
        self.temp_genres_list.column("Count", width=100, anchor=tk.CENTER)
        
        self.temp_genres_list.heading("Genre", text="Genre")
        self.temp_genres_list.heading("Count", text="Anime Count")
        
        # Add genres to the listbox
        for genre, count in genres:
            self.temp_genres_list.insert("", "end", values=(genre, count))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.temp_genres_list.yview)
        scrollbar.pack(side='right', fill='y')
        self.temp_genres_list.configure(yscrollcommand=scrollbar.set)
        
        # Button frame
        btn_frame = ttk.Frame(genre_window, padding="10")
        btn_frame.pack(fill='x')
        
        # Function to add selected genres
        def add_selected_genres():
            selected = [self.temp_genres_list.item(item)['values'][0] for item in self.temp_genres_list.selection()]
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
    
    def show_tree_menu(self, event):
        """Show context menu for search results tree"""
        # Select row under mouse
        iid = self.results_tree.identify_row(event.y)
        if iid:
            self.results_tree.selection_set(iid)
            self.tree_menu.post(event.x_root, event.y_root)
    
    def perform_search(self):
        """Execute search based on criteria"""
        # Clear existing results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Show loading dialog
        loading = LoadingDialog(self, message="Searching anime...")
        self.update_idletasks()
        
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
        
        # Function to run in thread
        def search_thread():
            # Perform search
            results = self.anime_data.search_anime(criteria)
            
            # Update GUI in main thread
            self.after(100, lambda: self.display_search_results(results, loading))
        
        # Start search in separate thread
        thread = threading.Thread(target=search_thread)
        thread.daemon = True
        thread.start()
    
    def display_search_results(self, results, loading_dialog):
        """Display search results and close loading dialog"""
        try:
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
        finally:
            # Close loading dialog
            loading_dialog.close()
    
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
    
    def recommend_similar(self):
        """Recommend anime similar to selected anime"""
        selected = self.results_tree.focus()
        if not selected:
            messagebox.showinfo("Select Anime", "Please select an anime first.")
            return
        
        # Get anime ID from selected row
        anime_id = self.results_tree.item(selected)['values'][0]
        anime_name = self.results_tree.item(selected)['values'][1]
        
        # Notify parent to switch tabs and show recommendations
        self.event_generate("<<ShowRecommendations>>", data=f"{anime_id}|{anime_name}")
    
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
            # Show loading dialog
            loading = LoadingDialog(self, message="Recording rating...")
            self.update_idletasks()
            
            # Function to run in thread
            def rate_thread():
                # Get user data reference
                user_data = self.recommendation_engine.user_data
                
                # Record the rating
                success = user_data.record_rating(anime_id, rating)
                
                # Update collaborative model if needed
                if success and len(user_data.user_ratings) >= 5:
                    self.recommendation_engine.train_collaborative_model()
                
                # Update GUI in main thread
                self.after(100, lambda: self.finish_rating(success, anime_name, rating, loading))
            
            # Start rating in separate thread
            thread = threading.Thread(target=rate_thread)
            thread.daemon = True
            thread.start()
    
    def finish_rating(self, success, anime_name, rating, loading_dialog):
        """Finish the rating process and close loading dialog"""
        try:
            if success:
                messagebox.showinfo("Rating Recorded", f"Your rating of {rating} for '{anime_name}' has been recorded.")
                # Notify that ratings have changed
                self.event_generate("<<RatingsChanged>>")
            else:
                messagebox.showerror("Error", "Could not record rating.")
        finally:
            # Close loading dialog
            loading_dialog.close()
    
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
        
        # Create a custom dialog
        details_window = tk.Toplevel(self)
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


class RecommendationsTab(ttk.Frame):
    """Tab for getting anime recommendations"""
    def __init__(self, parent, anime_data, recommendation_engine):
        super().__init__(parent, padding="10")
        self.anime_data = anime_data
        self.recommendation_engine = recommendation_engine
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the recommendations interface"""
        # Recommendation options
        options_frame = ttk.LabelFrame(self, text="Recommendation Options", padding="10")
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
        rec_results_frame = ttk.LabelFrame(self, text="Recommendations", padding="10")
        rec_results_frame.pack(fill='both', expand=True, pady=10)
        
        # Treeview for displaying recommendations
        self.rec_tree = ttk.Treeview(rec_results_frame, columns=("ID", "Name", "Genre", "Type", "Episodes", "Rating"))
        self.rec_tree.pack(side='left', fill='both', expand=True)
        
        # Configure tree columns
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
        feedback_frame = ttk.LabelFrame(self, text="Feedback", padding="10")
        feedback_frame.pack(fill='x', pady=10)
        
        ttk.Label(feedback_frame, text="Select an anime and provide feedback:").pack(anchor='w', padx=5, pady=5)
        
        feedback_btn_frame = ttk.Frame(feedback_frame)
        feedback_btn_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(feedback_btn_frame, text="👍 I like this", command=lambda: self.provide_feedback(True)).pack(side='left', padx=5)
        ttk.Button(feedback_btn_frame, text="👎 Not for me", command=lambda: self.provide_feedback(False)).pack(side='left', padx=5)
        ttk.Button(feedback_btn_frame, text="⭐ Rate", command=self.rate_recommendation).pack(side='left', padx=5)
        
        # Right-click menu for recommendation tree
        self.rec_menu = tk.Menu(self, tearoff=0)
        self.rec_menu.add_command(label="I Like This", command=lambda: self.provide_feedback(True))
        self.rec_menu.add_command(label="Not For Me", command=lambda: self.provide_feedback(False))
        self.rec_menu.add_command(label="Rate This Anime", command=self.rate_recommendation)
        self.rec_menu.add_separator()
        self.rec_menu.add_command(label="View Details", command=self.view_recommendation_details)
        
        # Bind right-click event
        self.rec_tree.bind("<Button-3>", self.show_rec_menu)
        self.rec_tree.bind("<Double-1>", lambda e: self.view_recommendation_details())
    
    def set_base_anime(self, anime_id, anime_name):
        """Set the base anime for recommendations"""
        self.base_anime_entry.delete(0, tk.END)
        self.base_anime_entry.insert(0, f"{anime_id} - {anime_name}")
        self.rec_type_var.set("content")
        
        # Automatically get recommendations
        self.get_recommendations()
    
    def browse_anime(self):
        """Browse anime to select one as a base for recommendations"""
        browse_window = tk.Toplevel(self)
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
        self.browse_anime_list = ttk.Treeview(list_frame, columns=("ID", "Name", "Genre", "Rating"), show="headings")
        self.browse_anime_list.pack(side='left', fill='both', expand=True)
        
        self.browse_anime_list.column("ID", width=50, anchor=tk.CENTER)
        self.browse_anime_list.column("Name", width=300)
        self.browse_anime_list.column("Genre", width=300)
        self.browse_anime_list.column("Rating", width=80, anchor=tk.CENTER)
        
        self.browse_anime_list.heading("ID", text="ID")
        self.browse_anime_list.heading("Name", text="Name")
        self.browse_anime_list.heading("Genre", text="Genre")
        self.browse_anime_list.heading("Rating", text="Rating")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.browse_anime_list.yview)
        scrollbar.pack(side='right', fill='y')
        self.browse_anime_list.configure(yscrollcommand=scrollbar.set)
        
        # Function to handle displaying browse results
        def display_browse_results(df_sorted, loading_dialog):
            try:
                # Populate the treeview
                for _, row in df_sorted.iterrows():
                    self.browse_anime_list.insert("", "end", values=(
                        row['anime_id'],
                        row['name'],
                        row['genre'],
                        row['rating']
                    ))
            finally:
                # Close loading dialog
                loading_dialog.close()
        
        # Function to search anime
        def search_anime():
            query = search_entry.get().lower()
            self.browse_anime_list.delete(*self.browse_anime_list.get_children())
            
            # Show loading dialog
            loading = LoadingDialog(browse_window, message="Searching anime...")
            browse_window.update_idletasks()
            
                            # Function to run in thread
            def search_thread():
                # Get top 100 anime by rating if no search term
                if not query:
                    df_sorted = self.anime_data.df.sort_values('rating', ascending=False).head(100)
                else:
                    df_sorted = self.anime_data.df[self.anime_data.df['name'].str.contains(query, case=False, na=False)]
                
                # Update GUI in main thread
                self.after(100, lambda: display_browse_results(df_sorted, loading))
            
            # Start search in separate thread
            thread = threading.Thread(target=search_thread)
            thread.daemon = True
            thread.start()
        
        # Search button
        search_btn = ttk.Button(search_frame, text="Search", command=search_anime)
        search_btn.pack(side='left', padx=5)
        
        # Button frame
        btn_frame = ttk.Frame(browse_window, padding="10")
        btn_frame.pack(fill='x')
        
        # Function to select an anime
        def select_anime():
            selected = self.browse_anime_list.focus()
            if selected:
                values = self.browse_anime_list.item(selected)['values']
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
    
    def show_rec_menu(self, event):
        """Show context menu for recommendations tree"""
        iid = self.rec_tree.identify_row(event.y)
        if iid:
            self.rec_tree.selection_set(iid)
            self.rec_menu.post(event.x_root, event.y_root)
    
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
        
        # Show loading dialog
        loading = LoadingDialog(self, message="Getting recommendations...")
        self.update_idletasks()
        
        # Function to run in thread
        def recommend_thread():
            # Get recommendations based on type
            if rec_type == "content" and anime_id:
                recommendations = self.recommendation_engine.get_content_recommendations(anime_id, top_n=num_recs)
            elif rec_type == "collaborative":
                recommendations = self.recommendation_engine.get_collaborative_recommendations(top_n=num_recs)
            else:  # hybrid or fallback
                recommendations = self.recommendation_engine.get_hybrid_recommendations(anime_id, top_n=num_recs)
            
            # Update GUI in main thread
            self.after(100, lambda: self.display_recommendations(recommendations, loading))
        
        # Start recommendation in separate thread
        thread = threading.Thread(target=recommend_thread)
        thread.daemon = True
        thread.start()
    
    def display_recommendations(self, recommendations, loading_dialog):
        """Display recommendations and close loading dialog"""
        try:
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
        finally:
            # Close loading dialog
            loading_dialog.close()
    
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
        
        # Show loading dialog
        loading = LoadingDialog(self, message="Recording feedback...")
        self.update_idletasks()
        
        # Function to run in thread
        def feedback_thread():
            # Record the feedback
            success = self.recommendation_engine.user_data.record_feedback(anime_id, liked, comment)
            
            # Update GUI in main thread
            self.after(100, lambda: self.finish_feedback(success, anime_name, liked, loading))
        
        # Start feedback in separate thread
        thread = threading.Thread(target=feedback_thread)
        thread.daemon = True
        thread.start()
    
    def finish_feedback(self, success, anime_name, liked, loading_dialog):
        """Finish the feedback process and close loading dialog"""
        try:
            if success:
                if liked:
                    messagebox.showinfo("Feedback Recorded", 
                                      f"You liked '{anime_name}'. This will help improve your recommendations.")
                else:
                    messagebox.showinfo("Feedback Recorded", 
                                      f"You didn't like '{anime_name}'. This will help improve your recommendations.")
                
                # Notify that ratings have changed
                self.event_generate("<<RatingsChanged>>")
            else:
                messagebox.showerror("Error", "Could not record feedback.")
        finally:
            # Close loading dialog
            loading_dialog.close()
    
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
            # Show loading dialog
            loading = LoadingDialog(self, message="Recording rating...")
            self.update_idletasks()
            
            # Function to run in thread
            def rate_thread():
                # Record the rating
                success = self.recommendation_engine.user_data.record_rating(anime_id, rating)
                
                # Record feedback based on rating
                if success:
                    liked = rating >= 7.0  # Consider 7+ as "liked"
                    self.recommendation_engine.user_data.record_feedback(anime_id, liked, f"Rated {rating}/10")
                
                # Update collaborative model if needed
                if success and len(self.recommendation_engine.user_data.user_ratings) >= 5:
                    self.recommendation_engine.train_collaborative_model()
                
                # Update GUI in main thread
                self.after(100, lambda: self.finish_rating(success, anime_name, rating, loading))
            
            # Start rating in separate thread
            thread = threading.Thread(target=rate_thread)
            thread.daemon = True
            thread.start()
    
    def finish_rating(self, success, anime_name, rating, loading_dialog):
        """Finish the rating process and close loading dialog"""
        try:
            if success:
                messagebox.showinfo("Rating Recorded", f"Your rating of {rating} for '{anime_name}' has been recorded.")
                # Notify that ratings have changed
                self.event_generate("<<RatingsChanged>>")
            else:
                messagebox.showerror("Error", "Could not record rating.")
        finally:
            # Close loading dialog
            loading_dialog.close()
    
    def view_recommendation_details(self):
        """View details of selected recommendation"""
        selected = self.rec_tree.focus()
        if not selected:
            messagebox.showinfo("Select Anime", "Please select an anime first.")
            return
        
        # Get anime details
        values = self.rec_tree.item(selected)['values']
        anime_id = values[0]
        anime_name = values[1]
        anime_genre = values[2]
        anime_type = values[3]
        anime_episodes = values[4]
        anime_rating = values[5]
        
        # Display details in a dialog (largely the same as view_anime_details)
        details_window = tk.Toplevel(self)
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