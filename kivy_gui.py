from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
import cv2
import numpy as np
from retinaface import Retinaface
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import os

prediction = None


def load(image_path='captured_photo.jpg'):
    global prediction
    model_test = AntiSpoofPredict(0, "./resources/anti_spoof_models")
    image_cropper = CropImage()
    frame = cv2.imread(image_path)  # 从拍摄的照片文件中读取图像
    image_bbox = model_test.get_bbox(frame)
    prediction = np.zeros((1, 3))
    test_speed = 0
    for model_name in os.listdir("./resources/anti_spoof_models"):
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


class FaceDetectionApp(App):
    def build(self):
        global prediction
        self.retinaface = Retinaface()
        self.model_dir = "./resources/anti_spoof_models"
        self.device_id = 0
        self.model_test = AntiSpoofPredict(self.device_id, self.model_dir)
        self.image_cropper = CropImage()
        self.prediction = prediction

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        self.layout = BoxLayout(orientation='vertical')
        self.camera = Camera(play=True, index=0)

        self.result_label = Label(text="Detection Result:")
        self.start_button = Button(text="Start Photo Detection")
        self.start_button.bind(on_release=self.start_detection)

        self.layout.add_widget(self.camera)
        self.layout.add_widget(self.result_label)
        self.layout.add_widget(self.start_button)

        return self.layout

    def start_detection(self, instance):
        frame = self.capture_frame()
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            if len(faces) > 0:
                cv2.imwrite("captured_photo.jpg", frame)
                print("Photo captured and saved as: captured_photo.jpg")

                result = self.silent_liveness_detection("captured_photo.jpg")

                if result == 1:
                    name = self.face_recognition("captured_photo.jpg")
                    self.result_label.text = f"Detection Result: Real Person, Name: {name}"
                else:
                    self.result_label.text = "Detection Result: Not a Real Person"
            else:
                self.result_label.text = "No faces detected"

    def capture_frame(self):
        frame = self.camera.texture
        if frame:
            frame_data = frame.pixels
            height = frame.height
            width = frame.width
            channels = 4  # Default for RGBA
            frame_array = np.frombuffer(frame_data, dtype='uint8')
            frame = frame_array.reshape((height, width, channels))
            return frame
        return None

    def silent_liveness_detection(self, image_path):
        # Implement silent liveness detection here, return 1 for real person, 0 for not a real person
        # Please add the relevant liveness detection logic
        frame = cv2.imread(image_path)
        image_bbox = self.model_test.get_bbox(frame)
        prediction = np.zeros((1, 3))

        for model_name in os.listdir(self.model_dir):
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
            img = self.image_cropper.crop(**param)

            predictions = self.model_test.predict_batch([img])
            prediction += predictions[model_name]

        label = np.argmax(prediction)
        if label == 1:
            return 1
        else:
            return 0

    def face_recognition(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print('Open Error! Try again!')
            return
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image, name = self.retinaface.detect_image(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if image_path != "":
                cv2.imwrite(image_path, image)
                print("Save processed image to the path: " + image_path)
                print("Name: " + name)
                return name


if __name__ == '__main__':
    load()
    FaceDetectionApp().run()
