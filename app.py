import sys
import time

import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# from color_detection import color_detection


class fashion_tools(object):
    def __init__(self, imageid, model, version=1.1):
        self.imageid = imageid
        self.model = model
        self.version = version

    def get_dress(self, stack=False):
        """limited to top wear and full body dresses (wild and studio working)"""
        """takes input rgb----> return PNG"""
        name = self.imageid
        file = cv2.imread(name)
        file = tf.image.resize_with_pad(file, target_height=512, target_width=512)
        rgb = file.numpy()
        file = np.expand_dims(file, axis=0) / 255.
        seq = self.model.predict(file)
        seq = seq[3][0, :, :, 0]
        seq = np.expand_dims(seq, axis=-1)
        seq[seq < 0.95] = 0
        seq[seq >= 0.95] = 1

        c1x = rgb * seq
        c2x = rgb * (1-seq)
        cfx = c1x + c2x

        mask = seq * 255.

        dummy = np.ones((rgb.shape[0], rgb.shape[1], 1))
        rgbx = np.concatenate((rgb, dummy*255), axis=-1)
        rgbs = np.concatenate((cfx, mask), axis=-1)

        if stack:
            stacked = np.hstack((rgbx, rgbs))
            return stacked
        else:
            return rgbs

    def get_patch(self):
        return None


if __name__ == '__main__':
    t0 = time.time()

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    pretrained_model = load_model("model/topwears.h5")

    input_img = sys.argv[1]
    output_path = sys.argv[2]

    api = fashion_tools(input_img, pretrained_model)
    output_img = api.get_dress(stack=False)

    cv2.imwrite(output_path, output_img)

    delta_t = time.time() - t0
    print('result has been processed in {} seconds and saved in {}!'.format(delta_t, output_path))
