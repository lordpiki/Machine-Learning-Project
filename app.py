import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image
import random
from ML import *
from img_manipulation import *

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing App")
        
        self.canvas_size = 280  # 10x scaling for better visibility
        self.pixel_size = 10
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=2)
        
        self.save_button = tk.Button(root, text="Save", command=self.save_drawing)
        self.save_button.grid(row=1, column=0, pady=10)
        
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=1, pady=10)
        
        self.number_label = tk.Label(root, text="Random Number: ", font=("Arial", 14))
        self.number_label.grid(row=2, column=0, columnspan=2)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.drawing = np.zeros((28, 28), dtype=np.uint8)
        

        
    
    def show_number(self, num):
        self.number_label.config(text=f"Prediction: {num}")

    
    def paint(self, event):
        x = event.x // self.pixel_size
        y = event.y // self.pixel_size
        if 0 <= x < 28 and 0 <= y < 28 and self.drawing[y, x] == 0:
            x1, y1 = x * self.pixel_size, y * self.pixel_size
            x2, y2 = x1 + self.pixel_size, y1 + self.pixel_size
            self.canvas.create_rectangle(x1 - self.pixel_size, y1 - self.pixel_size, 
                                         x2 + self.pixel_size, y2 + self.pixel_size, 
                                         fill="black", outline="black")
            self.drawing[y, x] = 255
            
            num = evaluateImage(self.get_current_image(), self.digit_network)
            self.show_number(num)
                
    def save_drawing(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            img = self.get_current_image()
            img.save(file_path)
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawing.fill(0)

    def get_current_image(self):
        # Create an RGBA image with black and alpha channel
        rgba_image = np.zeros((28, 28, 4), dtype=np.uint8)
        rgba_image[..., 3] = self.drawing  # Set alpha channel
        rgba_image[self.drawing == 255, :3] = 0  # Set black color for drawn pixels
        
        img = Image.fromarray(rgba_image, 'RGBA')
        return randomize_image(img, 0.2)

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
