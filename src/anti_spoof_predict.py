# anti_spoof_predict.py

import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F

from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}

class Detection:
    def __init__(self):
        caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "./resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height

        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3] * width, out[max_conf_index, 4] * height, \
                                   out[max_conf_index, 5] * width, out[max_conf_index, 6] * height
        bbox = [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]
        return bbox

class AntiSpoofPredict(Detection):
    def __init__(self, device_id, model_dir):
        super(AntiSpoofPredict, self).__init__()
        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")
        self.models = self.load_models(model_dir)

    def load_models(self, model_dir):
        models = {}
        for model_name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_name)
            h_input, w_input, model_type, _ = parse_model_name(model_name)
            kernel_size = get_kernel(h_input, w_input)
            model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            keys = iter(state_dict)
            first_layer_name = keys.__next__()
            if first_layer_name.find('module.') >= 0:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    name_key = key[7:]
                    new_state_dict[name_key] = value
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)
            models[model_name] = model
        return models

    def predict_batch(self, imgs):
        test_transform = trans.Compose([trans.ToTensor()])
        img_batch = torch.stack([test_transform(img) for img in imgs]).to(self.device)
        result_batch = {}

        for model_name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                result = model.forward(img_batch)
                result = F.softmax(result, dim=1).cpu().numpy()
                result_batch[model_name] = result

        return result_batch

