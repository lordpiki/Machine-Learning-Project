import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import random
from ML import *

class PhotoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Viewer")
        
        # Set initial window size and position
        self.root.geometry("700x600")  # Set initial dimensions (width x height)
        
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        self.random_number_label = tk.Label(self.root, font=("Helvetica", 24))
        self.random_number_label.pack(pady=10)

        select_button = tk.Button(self.root, text="Select Photo", command=self.select_photo, width=10, height=2, font=("Helvetica", 14))
        select_button.pack(pady=20)  # Increased padding around the button for better spacing
        
        self.digit_network = get_digit_model()

    def select_photo(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png")])
        if file_path:
            self.display_photo(file_path)

    def display_photo(self, file_path):
        image = Image.open(file_path)
        resized_image = image.resize((300, 300))  # Resize to 500x500 pixels
        photo = ImageTk.PhotoImage(resized_image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

        random_number = evaluateImage(image, self.digit_network)
        self.random_number_label.configure(text=f"Predicted number: {random_number}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoApp(root)
    root.mainloop()
