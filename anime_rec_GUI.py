import tkinter as tk
import pandas as pd
# !    this is where you put the path where anime.csv is !
csv_list_path = ("anime.csv")


with open(csv_list_path, 'r')as file:
    anime_record = pd.read_csv(csv_list_path)

root = tk.Tk()
root.title("Anime recommendation app")
w = 500 # width for the Tk root
h = 400 # height for the Tk root

# get screen width and height
ws = root.winfo_screenwidth() # width of the screen
hs = root.winfo_screenheight() # height of the screen

# ^^^^temporary calculate x and y coordinates for the Tk root window (where i want it open)
x = 940
y = 0
# ^^^^set the dimensions of the screen
# ^^^^and where it is placed!
root.geometry('%dx%d+%d+%d' % (w, h, 940, 0))

#labels for options
anime_name_label = tk.Label(text="Anime Name:")
anime_name_label.grid(row=0, column=0, sticky='w')
anime_genre_label = tk.Label(text="Anime Genre:")
anime_genre_label.grid(row=1, column=0, sticky='w')

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
    
    txt.delete(0.0, tk.END)
    txt.insert(tk.END, f"---- Here's what I found----\n")
    for i in anime_record.name:        
        if get_anime_from_entry in i:
            txt.insert(tk.END, f"{number}. {i}\n")
            number += 1  
    
#button to show the anime
show_anime_button = tk.Button(master=root, text="Show Anime List",command=show_anime_record_text_box)
show_anime_button.grid(row=5, column=0, sticky='w')

#button that shows anime by name
get_anime_by_name = tk.Button(master=root, text="Show Anime by Name", command=find_anime_by_name)
get_anime_by_name.grid(row=5, column=1)

root.mainloop()