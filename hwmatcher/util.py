import os
import sys
import cv2
from recordtype import recordtype
import matplotlib
from PIL import Image, ImageFont, ImageDraw
import numpy as np
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import patches
from .config import *


def add_paths():
    THIRD_PARTY_ROOT = "third-party"
    sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, "EAST"))
    sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, "hwnet"))


add_paths()
from east import EASTWrapper
from utils import HWNetInferenceEngine


class ComputeFeatures:
    def __init__(self, east_path=east_path, hwnet_path=hwnet_path):
        self.eastw = EASTWrapper(checkpoint_path=east_path)
        self.hwnet = HWNetInferenceEngine(hwnet_path)

    def __call__(self, image_A):
        Sample = recordtype("Sample", "image bboxes features")

        def sample(image):
            _image, unit_bboxes = self.eastw.predict(image)
            H, W, *_ = image.shape
            unit_bboxes = sorted(unit_bboxes, key=lambda b: b.y * W + b.x)
            return Sample(image=image, bboxes=unit_bboxes, features=None)

        a = sample(image_A)

        def f(tag, img, bboxes):
            for i, bbox in enumerate(bboxes):
                fpath = "data/intermediates/{}-{}.png".format(tag, i)
                cropped = img[bbox.y : bbox.Y, bbox.x : bbox.X, :]
                # print(fpath)
                cv2.imwrite(fpath, cropped)

        f("a", a.image, a.bboxes)

        a.features = self.hwnet(a)
        return a.features, a.bboxes


def plot(bboxes, savePath=None, color='r'):
    IMG_DIR = './images/'
    obj = {}
    images = []

    for bbox in bboxes:
        name = bbox[0]
        if name not in obj:
            obj[name] = []
        obj[name].append(bbox[1])

    for key, value in obj.items():
        images.append(key)
        img = cv2.imread(IMG_DIR+key)
        plt.figure(figsize=(21, 15), dpi=300)
        plt.imshow(img)
        axes = plt.gca()
        for bbox in value:
            x, y, X, Y = (
                    bbox['x'],
                    bbox['y'],
                    bbox['X'],
                    bbox['Y']
            )
            rect = patches.Rectangle(
                (x, y),
                (X-x+1), (Y-y+1),
                linewidth=1,
                edgecolor=color,
                facecolor='none'
            )
            axes.add_patch(rect)
        if savePath:
            plt.savefig(savePath + key)
    return images


def stringimage(string):
    img = Image.new('RGB', (200, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Comfortaa-Regular.ttf", 16)
    draw.text((5, 5), string, font=font, align="center", fill= (0, 0, 0))
    img = np.array(img)
    return img
