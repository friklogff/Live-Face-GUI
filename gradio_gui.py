"""

NAME : guitest

USER : admin

DATE : 17/10/2023

PROJECT_NAME : Silent-Face-Anti-Spoofing-master

CSDN : friklogff
"""
import cv2
import gradio as gr
import os

import cv2
import numpy as np
import gradio as gr
from retinaface2 import Retinaface
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# Initialize objects for face detection and anti-spoofing
retinaface = Retinaface()
model_dir = "./resources/anti_spoof_models"
device_id = 0
model_test = AntiSpoofPredict(device_id, model_dir)
image_cropper = CropImage()
prediction = None
def load(model_dir, device_id, image_path = 'captured_photo.jpg'):
    global prediction
    model_test = AntiSpoofPredict(device_id, model_dir)
    image_cropper = CropImage()
    frame = cv2.imread(image_path)  # 从拍摄的照片文件中读取图像
    image_bbox = model_test.get_bbox(frame)
    prediction = np.zeros((1, 3))
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        # 使用模型进行预测
        predictions = model_test.predict_batch([img])
        # 将当前模型的预测结果添加到总预测中
        prediction += predictions[model_name]
        break


def test_and_detect(image_path):
    global prediction

    frame = cv2.imread(image_path)
    image_bbox = model_test.get_bbox(frame)
    prediction = np.zeros((1, 3))

    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)

        predictions = model_test.predict_batch([img])
        prediction += predictions[model_name]

    label = np.argmax(prediction)
    if label == 1:
        name = detect_image(image_path, "processed_photo.jpg")
        return True,name
    else:
        return False,'UNREAL'


def detect_image(img, imgSAVE_path):
    image = cv2.imread(img)
    if image is None:
        print('Open Error! Try again!')
        return
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, name = retinaface.detect_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if imgSAVE_path != "":
            cv2.imwrite(imgSAVE_path, image)
            print("Save processed image to the path: " + imgSAVE_path)
            print("Name: " + name)
            return name


def capture_photo(img):
    """
    :param img:
    :return:
    """
    # if name == "":
    #     return "Name cannot be empty!"
    # print(img)
    if img is None:
        return "img cannot be empty"
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        face = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if len(face) > 0:  # 检查是否检测到人脸
            # 检测到人脸，立即拍照
            cv2.imwrite("tempx.jpg", img)

            flag, name = test_and_detect("tempx.jpg")
            return "Hello!,"+name,"tempx.jpg"
        else:
            cv2.imwrite("tempx.jpg", img)
            return 'NO FACE',"tempx.jpg"

demo = gr.Interface(
    capture_photo,
    [
        gr.components.Image(source="webcam", label="Webcam"),
    ],
    [
        gr.components.Text(label='output'),
        gr.components.Image(label="output")
    ],
    live=True,
)

if __name__ == '__main__':
    load(model_dir, device_id)

    demo.launch(share=True)
