import numpy as np
import cv2
from mlpipe.processors.i_processor import IPreProcessor


class PreProcessData(IPreProcessor):
    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        ground_truth = np.zeros(10)
        ground_truth[raw_data["label"]] = 1.0

        png_binary = raw_data["img"]
        png_img = np.frombuffer(png_binary, np.uint8)
        input_data = cv2.imdecode(png_img, cv2.IMREAD_COLOR)

        return raw_data, input_data, ground_truth, piped_params
