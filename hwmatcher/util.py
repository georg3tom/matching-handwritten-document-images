import os
import sys
import cv2
from recordtype import recordtype
from pathlib import Path
import torch
import matplotlib
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
from neigh_search import LSHIndex as LSH, L2ExactIndex as L2


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


