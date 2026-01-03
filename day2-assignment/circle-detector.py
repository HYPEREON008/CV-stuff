import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, ttk
# %%
input_dir = 'test-images'
output_dir = 'results'

# %%
def preprocess_image(image_path):
    ## load and preprocess image for circle detection
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0) # 0 is the sigma value
    

    return img, gray

# %%
def plot_image(img, output):
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Image')
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Detected Circles')
    plt.show()
    

# %%
def detect_circles(gray):
    ## detect circles using HoughCircles\
    circles = cv2.HoughCircles(gray, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1, 
                               minDist=20,
                               param1=param1_slider.get(),
                               param2=param2_slider.get(),
                               minRadius=0,
                               maxRadius=0)
    return circles[0] if circles is not None else []


# %%
def visualize_circles(img, circles, save_path=None):
    ## draw detected circles on the image
    output = img.copy()
    for (x, y, r) in circles:
        x = int(x)
        y = int(y)
        r = int(r)
        
        cv2.circle(output, (x, y), r, (0, 255, 0), 4) # circle outline
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1) # circle center
    
    #plot_image(img, output)

    if save_path:
        cv2.imwrite(save_path, output)
        print(f"Image with detected circles saved at: {save_path}")
    return output

# %%
def calculate_statistics(circles):
    ## calculate statistics of detected circles

    ## save statistics to a text file
    radii = [r for (x,y,r) in circles]
    if len(radii) == 0:
        return 0, 0, 0
    mean_radius = np.mean(radii)
    min_radius = np.min(radii)
    max_radius = np.max(radii)
    ## returns a dictionary of statistics
    stats = {
        "mean radius": mean_radius,
        "min radius": min_radius,
        "max radius": max_radius,
        "# circles": len(radii),
        
    }
    return stats

# %%
def main():
    img_path = input("Enter the file name of the image: ").strip()
    if not img_path:
        img_path = 'sample_image1.jpeg' #default image
        
    img_path = os.path.join(input_dir, img_path)
    
    img, preprocessed_img = preprocess_image(img_path)
    circles = detect_circles(preprocessed_img)
    output = visualize_circles(img, detect_circles(preprocessed_img), save_path = os.path.join(output_dir,'circles_' + os.path.basename(img_path)))
    plot_image(img, output)

    # write statistics to a text file
    stats = calculate_statistics(circles)
    stats_file = os.path.join(output_dir, 'statistics.txt')
    with open(stats_file, 'a') as f:
        f.write(f"Image: {os.path.basename(img_path)}\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
    print(f"Statistics saved at: {stats_file}")
    

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
    _, preprocessed_img = preprocess_image(img_path)
    circles = detect_circles(preprocessed_img)



    right_img = cv_to_tk(visualize_circles(cv2.imread(img_path), circles))

    right_img_label.config(image=right_img)
    right_img_label.image = right_img


input_dir = "test-images"
output_dir = "output-sketches"

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
    ## CLI verison
    # main()
    pass
