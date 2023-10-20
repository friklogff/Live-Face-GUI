import os
import torch
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from src.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}

class ModelLoader:
    def __init__(self, device_id, model_dir):
        self.device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
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
