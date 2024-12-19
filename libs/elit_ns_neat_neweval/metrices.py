import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def manhattan(data1, data2):

    if len(data1) != len(data2) or len(data1) < 1:
        return 0.0

    dist = np.sum(np.abs(data1 - data2))
    return dist


def euclidean(data1, data2):

    if len(data1) != len(data2) or len(data1) < 1:
        return 0.0

    dist = np.linalg.norm(data1 - data2)
    return dist

def weight_similarity(data1, data2):
    if len(data1) != len(data2) or len(data1) < 1:
        return 0.0

    data1 = np.array(data1)
    data2 = np.array(data2)
