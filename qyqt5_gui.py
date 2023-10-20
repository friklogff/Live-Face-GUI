"""

NAME : t5

USER : admin

DATE : 16/10/2023

PROJECT_NAME : Silent-Face-Anti-Spoofing-master

CSDN : friklogff
"""
import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap
from retinaface import Retinaface
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Initialize objects for face detection and anti-spoofing
        self.retinaface = Retinaface()
        self.model_dir = "./resources/anti_spoof_models"
        self.device_id = 0
        self.model_test = None
        self.image_cropper = CropImage()
        self.do_face_detection = False
        self.result_image_label = QLabel(self)
        self.result_image_label.setGeometry(50, 180, 400, 400)
        self.result_image_label.setScaledContents(True)

        self.next_button = QPushButton("Next", self)
        self.next_button.setGeometry(50, 600, 100, 50)
        self.next_button.clicked.connect(self.next_face_detection)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_result_image)

    def initUI(self):
        self.setWindowTitle("Face Recognition App")
        self.setGeometry(100, 100, 800, 700)

        self.start_button = QPushButton("Start Face Detection", self)
        self.start_button.setGeometry(50, 50, 200, 50)
        self.start_button.clicked.connect(self.start_face_detection)

        self.result_label = QLabel("Press 'F' to start face detection", self)
        self.result_label.setGeometry(50, 120, 400, 50)

    def start_face_detection(self):
        self.do_face_detection = True
        self.start_button.setDisabled(True)
        self.result_label.setText("Face detection in progress...")

        # Reset the result image label
        self.result_image_label.clear()

        self.next_face_detection()

    def next_face_detection(self):
        # Capture a photo
        self.take_photo("captured_photo.jpg")
        do_face_detection_result = self.test_and_detect("captured_photo.jpg")

        if do_face_detection_result == "FakeFace":
            self.result_label.setText("FakeFace detected. Press 'Next' to start face detection again.")
        else:
            self.result_label.setText(
                f"Hello! {do_face_detection_result}, RealFace detected. Press 'Next' to start face detection again.")

        self.next_button.setEnabled(True)

    def update_result_image(self):
        # Load and display the last processed image
        pixmap = QPixmap("captured_photo.jpg")
        self.result_image_label.setPixmap(pixmap)

    def take_photo(self, temp_img_path="captured_photo.jpg"):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Unable to open the camera")
            return

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Unable to get a frame")
                break

            cv2.imshow("Capture Photo", frame)

            key = cv2.waitKey(1)
            if key == 32:  # Space key
                break

            # Detect faces
            faces = self.detect_faces(frame)
            if len(faces) > 0:
                cv2.imwrite(temp_img_path, frame)
                print("Photo captured and saved as: " + temp_img_path)
                break

        cap.release()
        cv2.destroyAllWindows()

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
        return faces

    def load_anti_spoof_model(self):
        self.model_test = AntiSpoofPredict(self.device_id, self.model_dir)

    def test_and_detect(self, image_path):
        if self.model_test is None:
            self.load_anti_spoof_model()

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
        value = prediction[0][label] / 2
        if label == 1:
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
            name = self.detect_image(image_path, "processed_photo.jpg")
            self.result_label.setText(result_text + " - " + name)
            self.timer.start(1000)  # Update the result image every second
            return name
        else:
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
            self.result_label.setText(result_text)
            self.timer.start(1000)  # Update the result image every second
            return "FakeFace"

    def detect_image(self, img, temp_img_path):
        image = cv2.imread(img)
        if image is None:
            print('Open Error! Try again!')
            return
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            r_image, name = self.retinaface.detect_image(image)
            r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Processed Image", r_image)
            cv2.waitKey(0)
            if temp_img_path != "":
                cv2.imwrite(temp_img_path, r_image)
                print("Save processed image to the path: " + temp_img_path)
                print("Name: " + name)
                return name

if __name__ == "__main__":
    app = QApplication(sys.argv)
    face_recognition_app = FaceRecognitionApp()
    face_recognition_app.show()
    sys.exit(app.exec_())
