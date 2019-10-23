import numpy as np
from itertools import combinations


def unionf(categories):
    neurons = np.array([], dtype='int64')
    for i in categories:
        neurons = np.union1d(neurons, i)

    return neurons


def calMeanNodes(items):
    for num_elements in range(1, 11):
        I = list(combinations(items, num_elements))
        numNode = []
        for i in I:
            numNode.append(len(unionf(i)))
        numNode = np.array(numNode)
        print('Number of elements (%2d)' % num_elements)
        print('-> mean: %.1f'%(np.mean(np.array(numNode))+num_elements))

    return

