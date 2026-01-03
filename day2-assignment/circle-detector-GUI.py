import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, ttk
import circle_detector as cd ## import the circle detector program

input_dir = 'test-images'
output_dir = 'results'

def cv_to_tk(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (360, 360))
    h, w, _ = img.shape
    ppm = f"P6 {w} {h} 255 ".encode() + img.tobytes()
    return tk.PhotoImage(data=ppm, format="PPM")

def select_image():
    global img_path

    img_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )

    if not img_path:
        return

    original = cv2.imread(img_path)
    tk_img = cv_to_tk(original)

    left_img_label.config(image = tk_img)
    left_img_label.image = tk_img

def action():
    if not img_path:
        return
    _, preprocessed_img = cd.preprocess_image(img_path)
    circles = cd.detect_circles(preprocessed_img, param1=param1_slider.get(), param2=param2_slider.get())

    right_img = cv_to_tk(cd.visualize_circles(cv2.imread(img_path), circles))

    right_img_label.config(image=right_img)
    right_img_label.image = right_img


img_path = None


root = tk.Tk()
color_mode = tk.BooleanVar(value=False)
root.title("Circle Detector")

root.geometry("900x520")
root.resizable(False, False)

# Main container
Main = ttk.Frame(root, padding=15)
Main.pack(fill="both", expand=True)

# Top control frame
top = ttk.Frame(Main)
top.pack(pady = 10)

ttk.Button(top, text="Select Image", command = select_image).grid(row=0, column=0, padx=10)
ttk.Label(top, text="Param 1").grid(row=0, column=1, padx=10)
ttk.Label(top, text="Param 2").grid(row=0, column=2, padx=10)

param1_slider = tk.Scale(top, from_=3, to=101, orient="horizontal", resolution=1)
param1_slider.set(21)
param1_slider.grid(row=1, column=1, padx=10)

param2_slider = tk.Scale(top, from_=3, to=101, orient="horizontal", resolution=1)
param2_slider.set(21)
param2_slider.grid(row=1, column=2, padx=10)

ttk.Button(top, text="Detect", command = action).grid(row=0, column=3, padx=10)
ttk.Checkbutton(top, text="Color Sketch", variable = color_mode).grid(row=0, column=4, padx=10)

# Image display frame
img_frame = ttk.Frame(Main)
img_frame.pack(pady=10)

default_original_img = tk.PhotoImage(height=360, width=360)  # Placeholder for original image
default_sketch_img = tk.PhotoImage(height=360, width=360)  # Placeholder for detected circles image

left_img_label = tk.Label(img_frame, text="Original Image", background="lightgray", compound="top", image=default_original_img)
left_img_label.grid(row=1, column=0, padx=10)

right_img_label = tk.Label(img_frame, text="Detected Circles", background="lightgray", compound="top", image=default_sketch_img)
right_img_label.grid(row=1, column=1, padx=10)

root.mainloop()

if __name__ == "__main__":
    ## GUI version
    pass
