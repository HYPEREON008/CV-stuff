import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, ttk

def pencil_sketch(img_path, blur_ksize=21):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    #kernel size should be odd and greater than 1
    #no need to check here as user will provide correct value from the slider
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert color of image from BGR to GRAY
    img_invert = cv2.bitwise_not(img_gray) #invert the gray image
    img_smooth = cv2.GaussianBlur(img_invert, (blur_ksize, blur_ksize), 0) #cv2.GaussianBlur(src, ksize, sigmaX) sigmaX is standard deviation in X direction, here 0 means calculated from kernel size

    #now invert the blurred image
    img_smooth_invert = cv2.bitwise_not(img_smooth)
    #create the pencil sketch image:
    img_pencil_sketch = cv2.divide(img_gray, img_smooth_invert, scale = 256)
    if not color_mode.get():
        return img, img_pencil_sketch
    
    #create color pencil sketch
    img_color_pencil_sketch = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(img_pencil_sketch))
    threshold = 1  # adjust
    mask = cv2.inRange(img_color_pencil_sketch, (0, 0, 0), (threshold, threshold, threshold))
    img_color_pencil_sketch[mask > 0] = (255, 255, 255)

    return (img, img_color_pencil_sketch)


def display_img(original, sketch, save_path=None):
    # plt.figure(figsize = (10,10)) #set figure size to 10x10 inches
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Original Image')
    plt.subplot(1,2,2)
    plt.imshow(sketch, cmap = 'gray')
    plt.axis('off')
    plt.title('Pencil Sketch Image')
    plt.show()

    if save_path:
        cv2.imwrite(save_path, sketch)
        print(f"Pencil sketch image saved at: {save_path}")


def main():
    img_path = input("Enter the file name of the image: ").strip()
    if not img_path:
        img_path = 'sample_image1.jpeg'
        
    img_path = os.path.join(input_dir, img_path)
    print(f"Processing image: {img_path}")
    original, sketch = pencil_sketch(img_path)
    display_img(original, sketch, save_path = os.path.join(output_dir, 'color_'*color_mode+'pencil_sketch_'+ os.path.basename(img_path)))

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

def convert_image():
    if not img_path:
        return
    _, sketch = pencil_sketch(img_path, blur_ksize=kernel_slider.get())

    right_img = cv_to_tk(sketch)

    right_img_label.config(image=right_img)
    right_img_label.image = right_img


input_dir = "test-images"
output_dir = "output-sketches"

img_path = None


root = tk.Tk()
color_mode = tk.BooleanVar(value=False)
root.title("Pencil Sketch Converter")

root.geometry("900x520")
root.resizable(False, False)

# Main container
Main = ttk.Frame(root, padding=15)
Main.pack(fill="both", expand=True)

# Top control frame
top = ttk.Frame(Main)
top.pack(pady = 10)

ttk.Button(top, text="Select Image", command = select_image).grid(row=0, column=0, padx=10)
ttk.Label(top, text="Kernel Size for Blur:").grid(row=0, column=1, padx=10)

kernel_slider = tk.Scale(top, from_=3, to=101, orient="horizontal", resolution=2)
kernel_slider.set(21)
kernel_slider.grid(row=0, column=2, padx=10)

ttk.Button(top, text="Convert", command = convert_image).grid(row=0, column=3, padx=10)
ttk.Checkbutton(top, text="Color Sketch", variable = color_mode).grid(row=0, column=4, padx=10)

# Image display frame
img_frame = ttk.Frame(Main)
img_frame.pack(pady=10)

default_original_img = tk.PhotoImage(height=360, width=360)  # Placeholder for original image
default_sketch_img = tk.PhotoImage(height=360, width=360)  # Placeholder for sketch image

left_img_label = tk.Label(img_frame, text="Original Image", background="lightgray", compound="top", image=default_original_img)
left_img_label.grid(row=1, column=0, padx=10)

right_img_label = tk.Label(img_frame, text="Pencil Sketch", background="lightgray", compound="top", image=default_sketch_img)
right_img_label.grid(row=1, column=1, padx=10)

root.mainloop()

if __name__ == "__main__":
    ## CLI verison
    # main()
    pass
