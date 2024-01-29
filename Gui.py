import numpy as np
import cv2
import tensorflow as tf
from tkinter import filedialog
from tkinter import Tk, Label, Button, Canvas
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input


def image_enhance(image):
    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    enhanced_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    return enhanced_image

def denoise_image(image):

    denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised_image


def region_segmentation(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded


model = tf.keras.models.load_model('breast_cancer_detection_model.h5')


def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def process_and_predict(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色空间


    image = image_enhance(image)
    image = denoise_image(image)
    segmented = region_segmentation(image)


    preprocessed_image = preprocess_image(image_path)


    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)


    result = "WARNING: Potential signs of breast cancer are detected!" \
        if predicted_class[0] == 1 else "Everything normal。"
    return result, segmented



root = Tk()
root.title("Breast Cancer Detection")

def upload_image():
    global img_label, processed_img_label
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        img_label.configure(image=img)
        img_label.image = img


        result, segmented = process_and_predict(file_path)
        result_label.configure(text=result)


        segmented_pil = Image.fromarray(segmented)
        segmented_pil = segmented_pil.resize((250, 250), Image.Resampling.LANCZOS)
        segmented_photo = ImageTk.PhotoImage(segmented_pil)
        processed_img_label.configure(image=segmented_photo)
        processed_img_label.image = segmented_photo


upload_btn = Button(root, text="Upload images", command=upload_image)
upload_btn.pack()


img_label = Label(root)
img_label.pack(side="left", padx=10)

processed_img_label = Label(root)
processed_img_label.pack(side="right", padx=10)


result_label = Label(root, text="The results will be shown here")
result_label.pack()


root.mainloop()
