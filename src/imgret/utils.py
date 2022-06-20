import random
from enum import Enum
from json import JSONEncoder

import cv2 as cv
import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class Distances(str, Enum):
    cityblock = "cityblock"
    cosine = "cosine"
    euclidean = "euclidean"
    chebysev = "chebysev"
    dice = "dice"
    jaccard = "jaccard"
    canberra = "canberra"
    sqeuclidean = "sqeuclidean"


def get_image_features(image):
    # Instantiate the ORB object
    # Limit the number of features so we can have same size vector descriptors
    orb = cv.ORB_create()

    # Detect the keypoints
    keypoints = orb.detect(image, None)

    # compute the descriptors with ORB
    _, descriptor = orb.compute(image, keypoints)
    features = descriptor.flatten()

    return features


def coin_toss(p):
    return random.random() < p
