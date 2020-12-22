import os
import sys
import cv2
from PIL import Image, ImageFont, ImageDraw
import faiss
import numpy as np
from recordtype import recordtype
import pickle
from os import path


def add_paths():
    THIRD_PARTY_ROOT = 'third-party'
    sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, 'EAST'))
    sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, 'hwnet'))
    sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, 'neigh_search'))

add_paths()

from east import EASTWrapper
from utils import HWNetInferenceEngine
import torch
from torch.nn.modules.distance import CosineSimilarity, PairwiseDistance
from munkres import Munkres, make_cost_matrix
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import patches
from neigh_search import LSHIndex as LSH, L2ExactIndex as L2

class ComputeFeatures:
    def __init__(self, east_path, hwnet_path):
        self.eastw = EASTWrapper(checkpoint_path=east_path)
        self.hwnet = HWNetInferenceEngine(hwnet_path)

    def __call__(self, image_A):
        Sample = recordtype('Sample', 'image bboxes features')
        def sample(image):
            _image, unit_bboxes = self.eastw.predict(image)
            H, W, *_ = image.shape
            unit_bboxes = sorted(unit_bboxes, key=lambda b: b.y * W + b.x) 
            return Sample(image=image, bboxes=unit_bboxes, features=None)

        a = sample(image_A)

        def f(tag, img, bboxes):
            for i, bbox in enumerate(bboxes):
                fpath = 'data/intermediates/{}-{}.png'.format(tag, i)
                cropped = img[bbox.y:bbox.Y, bbox.x:bbox.X, :]
                # print(fpath)
                cv2.imwrite(fpath, cropped)

        f("a", a.image, a.bboxes)

        a.features = self.hwnet(a)
        return a.features

    def debug_nn(self, matrix):
        nA, nB = matrix.shape
        best = []
        for i in range(nA):
            js = np.argsort(matrix[i, :])
            #  print(matrix[i, js])
            best.append((i, js[0]))
        # print(len(best))
        xs, ys = list(zip(*best))
        # print(set(xs))
        # print(set(ys))
        return best

def drawQ(bboxes, savePath,color= 'r'):
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
        plt.savefig(savePath + key)
    return images

def stringQuery(query = "EfficientDet"):
    east_path = './east_icdar2015_resnet_v1_50_rbox/'
    hwnet_path = os.path.join('third-party/hwnet/pretrained', 'iam-morph0.t7')

    index_path = "./index"
    labels_path = "./labels.npy"
    obj_path = "./obj.pkl"
    imsave_loc = "./webapp/app/static/images/"

    labels = np.load(labels_path)
    index = faiss.read_index(index_path)
    with open(obj_path, 'rb') as f:
        obj = pickle.load(f)

    img = Image.new('RGB', (200, 100), color = (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Comfortaa-Regular.ttf", 16)
    draw.text((5, 5), query, font = font, align ="center",fill= (0, 0, 0))
    img = np.array(img)

    compare = ComputeFeatures(east_path, hwnet_path)
    features = compare(img)
    features = features.numpy()

    distances, indices = index.search(features, 10)
    bbox = []
    for indx,i in enumerate(indices[0]):
        bbox.append(obj[i])
    images = drawQ(bbox, imsave_loc)
    return images, distances[0]

if __name__ == '__main__':
    east_path = './east_icdar2015_resnet_v1_50_rbox/'
    hwnet_path = os.path.join('third-party/hwnet/pretrained', 'iam-morph0.t7')
    imsave_loc = "./"

    index_path = "./index"
    labels_path = "./labels.npy"
    obj_path = "./obj.pkl"

    labels = np.load(labels_path)
    index = faiss.read_index(index_path)
    with open(obj_path, 'rb') as f:
        obj = pickle.load(f)

    # query = input("QUERY word:")
    query = "Introduction"
    img = Image.new('RGB', (200, 100), color = (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Comfortaa-Regular.ttf", 16)
    draw.text((5, 5), query, font = font, align ="center",fill= (0, 0, 0))
    img = np.array(img)

    compare = ComputeFeatures(east_path, hwnet_path)
    features = compare(img)
    features = features.numpy()

    distances, indices = index.search(features, 10)
    print(indices)
    bbox = []
    for indx,i in enumerate(indices[0]):
        bbox.append(obj[i])
    drawQ(bbox, imsave_loc)

    # print(labels[np.array(indices)], distances)
