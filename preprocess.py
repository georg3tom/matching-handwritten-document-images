import sys
from pathlib import Path
import cv2
import torch
sys.path.insert(0, './')
from hwmatcher.util import ComputeFeatures
from hwmatcher.config import IMG_DIR
from hwmatcher.neigh_search import LSHIndex as LSH, L2ExactIndex as L2

if __name__ == "__main__":
    compute = ComputeFeatures()

    vectors = None
    labels = []
    files = list(Path(IMG_DIR).glob("*.png"))
    files = files[:2]
    cur = 0
    obj = {}
    for f in files:
        print(f.name)
        image = cv2.imread(IMG_DIR + f.name)
        feats, bbox = compute(image)
        if vectors is None:
            vectors = feats.clone().detach()
        else:
            vectors = torch.cat([vectors, feats], dim=0)

        for b in bbox:
            tmp = {
                "x": b.x,
                "y": b.y,
                "X": b.X,
                "Y": b.Y,
            }
            obj[cur] = [f.name, tmp]
            labels.append(cur)
            cur = cur + 1

    vectors = vectors.numpy()
    lsh = LSH(vectors, labels, obj)
    lsh.build(num_bits=128)
    lsh.write()
