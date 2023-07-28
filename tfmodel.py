import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2

class TFTagger:
    def __init__(self, model_directory,batch_size=8):
        self.model=load_model(model_directory)
        # Get the input tensor(s) of the model
        input_tensors = self.model.input
        # Extract batch size, height, and width from the input tensor(s) shape
        self.batch_size = batch_size
        self.height = input_tensors.shape[1]
        self.width = input_tensors.shape[2]
        self.dtype = np.float32

    def __call__(self, pre_processed_img_list):
        post_processed_img_list = []
        for img in pre_processed_img_list:
            h, w, c = img.shape
            mx = max(h, w)
            scale_x = self.width / mx
            scale_y = self.height / mx
            scale = min(scale_y, scale_y)
            canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
            new_h = int(scale * h)
            new_w = int(scale * w)
            canvas[0:new_h, 0:new_w] = cv2.resize(img, (new_w, new_h))
            shift_y = int((self.height - new_h) / 2)
            shift_x = int((self.width - new_w) / 2)
            canvas = np.roll(canvas, shift_y, axis=0)
            canvas = np.roll(canvas, shift_x, axis=1)
            post_processed_img_list += [canvas]
        x = np.array(post_processed_img_list, dtype=self.dtype)
        results=self.model.predict(x,verbose=0)

        return results


r"""from glob import glob

filelist=glob(r'D:\datasetstosend\kagome\10_A/*.png')
filelist=filelist[0:8]
imglist=[cv2.imread(file) for file in filelist]
tagger=TFTagger(r'W:\python\utils\autotagger\wd-v1-4-swinv2-tagger-v2')

results=tagger(imglist)
p=0"""