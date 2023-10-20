import wx
import cv2
import numpy as np
from retinaface import Retinaface
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import os

retinaface = Retinaface()
model_dir = "./resources/anti_spoof_models"
device_id = 0
model_test = AntiSpoofPredict(device_id, model_dir)
image_cropper = CropImage()
prediction = None
class FaceDetectionApp(wx.Frame):
    def __init__(self):
        super().__init__(None, title="人脸检测应用", size=(600, 400))
        self.retinaface = Retinaface()
        self.model_dir = "./resources/anti_spoof_models"
        self.device_id = 0
        self.model_test = AntiSpoofPredict(self.device_id, self.model_dir)
        self.image_cropper = CropImage()
        self.prediction = None

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        panel = wx.Panel(self)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.start_button = wx.Button(panel, label="开始拍照检测")
        self.result_label = wx.StaticText(panel, label="检测结果：")
        self.image_ctrl = wx.StaticBitmap(panel, size=(300, 300))

        self.start_button.Bind(wx.EVT_BUTTON, self.start_detection)

        hbox.Add(self.image_ctrl, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        vbox.Add(self.start_button, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        vbox.Add(self.result_label, 0, wx.ALIGN_CENTER | wx.ALL, 5)

        hbox.Add(vbox, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        panel.SetSizer(hbox)

        self.cap = cv2.VideoCapture(0)  # 打开默认摄像头
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update_camera, self.timer)
        self.timer.Start(10)  # 更新频率（毫秒）

    def update_camera(self, event):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            img = wx.Image(width, height, frame.tobytes())
            bmp = wx.Bitmap(img)
            self.image_ctrl.SetBitmap(bmp)

    def start_detection(self, event):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))

            if len(faces) > 0:  # 如果检测到人脸
                cv2.imwrite("captured_photo.jpg", frame)  # 拍照并保存为 "captured_photo.jpg"
                print("已拍照并保存为：captured_photo.jpg")

                result = self.silent_liveness_detection("captured_photo.jpg")

                if result == 1:
                    name = self.face_recognition("captured_photo.jpg")
                    self.result_label.SetLabel(f"检测结果：真人，姓名：{name}")
                else:
                    self.result_label.SetLabel("检测结果：非真人")
            else:
                self.result_label.SetLabel("未检测到人脸")

    def silent_liveness_detection(self, image_path):
        # 在这里执行静默活体检测，返回1表示真人，0表示非真人
        # 请添加相应的活体检测逻辑
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
            # name = self.face_recognition(image_path)
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
            image, name = retinaface.detect_image(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if image_path != "":
                cv2.imwrite(image_path, image)
                print("Save processed image to the path: " + image_path)
                print("Name: " + name)
                return name

if __name__ == '__main__':
    app = wx.App()
    frame = FaceDetectionApp()
    frame.Show()
    app.MainLoop()
