import pickle
import faiss
from hwmatcher.util import plot
from hwmatcher.util import stringimage
from hwmatcher.util import ComputeFeatures

class stringQuery:
    def __init__(self):
        self.compute = ComputeFeatures()
        index_path = "./index"
        obj_path = "./obj.pkl"
        with open(obj_path, 'rb') as f:
            self.obj = pickle.load(f)
        self.index = faiss.read_index(index_path)

    def query(self, string, savePath=None):
        img = stringimage(string)
        features, _ = self.compute(img)
        features = features.numpy()
        distances, indices = self.index.search(features, 10)
        bbox = []
        for indx, i in enumerate(indices[0]):
            bbox.append(self.obj[i])
        images = plot(bbox, savePath)
        return images, distances[0]

