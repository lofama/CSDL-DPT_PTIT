import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import HienThi49Anh
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg; *.jpeg; *.png")])
    if file_path:
        display_images(file_path)
def hienThi49():
    HienThi49Anh.display_images_in_grid()
def display_images(file_path):
    # Load the image
    image = Image.open(file_path)

    # Resize the image to fit one-fourth of the original size
    width, height = image.size
    new_width = int(width / 4)
    new_height = int(height / 4)
    resized_image = image.resize((new_width, new_height))

    # Create Tkinter image objects
    tk_image1 = ImageTk.PhotoImage(resized_image)
    tk_image2 = ImageTk.PhotoImage(resized_image)
    tk_image3 = ImageTk.PhotoImage(resized_image)

    # Display images
    label1.config(image=tk_image1)
    label2.config(image=tk_image2)
    label3.config(image=tk_image3)

    # Keep references to avoid garbage collection
    label1.image = tk_image1
    label2.image = tk_image2
    label3.image = tk_image3
    
    # Display image name
    file_name = os.path.basename(file_path)
    image_name_label1.config(text="img_data/1.jpg")
    image_name_label2.config(text="img_data/2.jpg")
    image_name_label3.config(text="img_data/3.jpg")
# Create the main application window
root = tk.Tk()
root.title("Image Viewer")

# Create labels for displaying images
label1 = tk.Label(root, text="")
label1.grid(row=0, column=0)

label2 = tk.Label(root)
label2.grid(row=0, column=1)

label3 = tk.Label(root)
label3.grid(row=0, column=2)
# Create label for displaying image name
image_name_label1 = tk.Label(root, text="")
image_name_label1.grid(row=1, column=0)
image_name_label2 = tk.Label(root, text="")
image_name_label2.grid(row=1, column=1)
image_name_label3 = tk.Label(root, text="")
image_name_label3.grid(row=1, column=2)
# Create a canvas to display images
canvas = tk.Canvas(root)
canvas.grid(row=2, column=0, columnspan=3, sticky="nsew")

# # Create a scrollbar
# scrollbar = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
# scrollbar.grid(row=3, column=0, columnspan=3, sticky="ew")

# # Configure canvas
# canvas.config(xscrollcommand=scrollbar.set)

# Create and place button for opening image
open_button = tk.Button(root, text="Find Image", command=open_image)
open_button.grid(row=3, column=1)
open_button = tk.Button(root, text="Show Image", command=hienThi49)
open_button.grid(row=3, column=2)

# Start the Tkinter event loop
root.mainloop()
