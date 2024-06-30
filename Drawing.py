import tkinter as tk
from tkinter import filedialog
import cv2
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
        self.pixel_size = 1
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=2)
        
        self.save_button = tk.Button(root, text="Save", command=self.save_drawing)
        self.save_button.grid(row=1, column=0, pady=10)
        
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=1, pady=10)
        
        self.number_label = tk.Label(root, text="Random Number: ", font=("Arial", 14))
        self.number_label.grid(row=2, column=0, columnspan=2)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.drawing = np.zeros((280, 280), dtype=np.uint8)
        
        self.digit_network = get_digit_model()

        self.radius = 7
        
    
    def show_number(self, num):
        self.number_label.config(text=f"Prediction: {num}")

    
    def paint(self, event):
        x = event.x // self.pixel_size
        y = event.y // self.pixel_size
        
        x, y = max(1, x), max(1, y)
        x, y = min(270, x), min(270, y)
        x1, y1 = x * self.pixel_size, y * self.pixel_size
        x2, y2 = x1 + self.pixel_size, y1 + self.pixel_size
        self.drawing[y, x] = 255
        self.drawing = cv2.circle(self.drawing, (x, y), self.radius, 255, -1)
        
        self.canvas.create_oval(x1 - self.radius, y1 - self.radius, x2 + self.radius, y2 + self.radius, outline="black", fill="black")
        
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
        rgba_image = np.zeros((280, 280, 4), dtype=np.uint8)
        rgba_image[..., 3] = self.drawing  # Set alpha channel
        rgba_image[self.drawing == 255, :3] = 0  # Set black color for drawn pixels
        
        img = Image.fromarray(rgba_image, 'RGBA')
        img = img.resize((28, 28))
        return img
        # return randomize_image(img, 0.2)

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
