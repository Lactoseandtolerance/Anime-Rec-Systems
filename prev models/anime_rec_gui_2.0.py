import tkinter as tk
import pandas as pd
# !    this is where you put the path where anime.csv is !
csv_list_path = ("anime.csv")

with open(csv_list_path, 'r')as file:
    anime_record = pd.read_csv(csv_list_path)

root = tk.Tk()
root.title("Anime recommendation app")
w = 600 # width for the Tk root
h = 500 # height for the Tk root

# get screen width and height
ws = root.winfo_screenwidth() # width of the screen
hs = root.winfo_screenheight() # height of the screen

# ^^^^temporary calculate x and y coordinates for the Tk root window (where i want it open)
x = 840
y = 0
# ^^^^set the dimensions of the screen
# ^^^^and where it is placed!
root.geometry('%dx%d+%d+%d' % (w, h, x, y))

#labels for options
anime_name_label = tk.Label(text="Anime Name:")
anime_name_label.grid(row=0, column=0, sticky='w')

#function to display info for anime name
def find_anime_by_name():
    get_anime_from_entry = anime_name_entry.get()

    txt.delete(1.0, tk.END)  

    if any(anime_record['name'].str.contains(get_anime_from_entry, case=False)):
        txt.insert(tk.END, f"---- Here's what I found ----\n")

        for index, row in anime_record.iterrows():
            anime_name = row['name']
            anime_genre = row['genre']
            anime_rating = row['rating']

            if get_anime_from_entry.lower() in anime_name.lower():
                txt.insert(tk.END, f"Name: {anime_name}\nGenre: {anime_genre}\nRating: {anime_rating}\n\n")
    else:
        txt.insert(tk.END, "No matching anime found.")

#search button for anime names
search_anime_button = tk.Button(master=root, text="Search",command=find_anime_by_name)
search_anime_button.grid(row=0, column=3, sticky='w')

#Labels for options
anime_genre_label = tk.Label(text="Anime Genre:")
anime_genre_label.grid(row=1, column=0, sticky='w')

#function for searching genres even if user doesn't type genres on correct order
def find_anime_by_genre():

    anime_genre_entry_on_a_list = anime_genre_entry.get() #get info from anime genre box
    anime_genre_from_csv = anime_record.genre #genres only from csv file

    #turning the two variables above into lists so is easier to compare them
    anime_genre_from_csv_on_a_list = [] 
    anime_genre_entry_on_a_list_list = [] 

    for i in anime_genre_entry_on_a_list.split(','): #first get rid of commas
        removed_whitespace = i.strip().title() #then get rid of whitespace and make the first letter of the genre into an uppercase
        anime_genre_entry_on_a_list_list.append(removed_whitespace)#add the word to the list

    for i in anime_genre_from_csv: #for every genre in the csv add it to the genre csv list
        anime_genre_from_csv_on_a_list.append(f"{i}\n")

    txt.delete(0.0, tk.END) #refreshes the text box

    for index, row in anime_record.iterrows():
        anime_name = row['name'] #get name from current row
        anime_genre = row['genre'] #get genre from current row
        anime_rating = row['rating'] #get rating from current row 
        
        count = 0 #count to only show anime with the genres user wants

        for j in range(len(anime_genre_entry_on_a_list_list)): #iterate through the whole list to check for genres
            if anime_genre_entry_on_a_list_list[j] in anime_genre: #if the genre from the text box is in the genres from the csv
                count += 1#add to count
        if count == len(anime_genre_entry_on_a_list_list):#if the count is the same as the length of genres from genre entry
            txt.insert(tk.END, f"Name: {anime_name}\nGenre: {anime_genre}\nRating: {anime_rating}\n\n")#then show info on text box
    else:
        txt.insert(tk.END, "No matching anime found.")
        
#search button for anime genres
search_genre_button = tk.Button(master=root, text="Search",command=find_anime_by_genre)
search_genre_button.grid(row=1, column=3, sticky='w')

#text box entries
anime_name_entry = tk.Entry(master=root)
anime_name_entry.grid(row=0, column=1)
anime_genre_entry = tk.Entry(master=root)
anime_genre_entry.grid(row=1, column=1)

#create a text widget to display anime records
txt = tk.Text(master=root, height=10, width=60)
txt.grid(row=4, column=0, columnspan=2)

#function that shows first 100 anime on list
def show_anime_record_text_box():
    anime_names = anime_record.name
                                                    
    for i in anime_names[:5]:
        txt.insert(tk.END, f"{i}\n")

#function that inputs anime that contains a word from name entry box into text box
def find_anime_by_name():
    get_anime_from_entry = anime_name_entry.get()

    number= 1
                                                    #might delete
    txt.delete(0.0, tk.END)
    txt.insert(tk.END, f"---- Here's what I found----\n")
    for i in anime_record.name:        
        if get_anime_from_entry in i:
            txt.insert(tk.END, f"{number}. {i}\n")
            number += 1  

#function that shows all genres
def show_anime_genres_text_box():
    txt.delete(0.0, tk.END)
    txt.insert(tk.END, f"---- Here's what I found----\n")
    unique_entries_list = {}
    unique_entries_list = anime_record['genre'].str.split(',').explode().str.strip().unique().tolist()
    
    for i in unique_entries_list:
        txt.insert(tk.END, f"{i}\n")

#button to show the anime
show_anime_button = tk.Button(master=root, text="Show Anime List",command=show_anime_record_text_box)
show_anime_button.grid(row=5, column=0, sticky='w')

#button that shows anime by name
get_anime_by_name = tk.Button(master=root, text="Show Anime by Name", command=find_anime_by_name)
get_anime_by_name.grid(row=5, column=1)

#button to print all genres
show_genres_button = tk.Button(master=root, text="Show Genres List",command=show_anime_genres_text_box)
show_genres_button.grid(row=5, column=3, sticky='w')

root.mainloop()
