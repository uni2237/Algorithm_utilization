import random
import numpy as np
from numpy import argmax

def greedy():
    score=1
    a=0
    data=np.zeros((10,5))
    for i in range(10):
        np.random.seed(score)
        for j in range(5):
            data[i][j]=np.random.random()
        x=argmax(data[i])
        a+=data[i][x]
        score=int((data[i][x])*10)*score
    return [argmax(s) for s in data],a

def beam():
    a=0
    score = 1
    data = np.zeros((10, 5))
    k = 3
    np.random.seed(score)
    for j in range(5):
        data[0][j] = np.random.random()  # 0
    order = sorted(data[0], reverse=True)
    score = order[:k]

    for i, s in enumerate(score):

        np.random.seed(int(s * 10))
        for j in range(5):
            data[i + 1][j] = np.random.random()  # 1,2,3
        score = sorted(data[i + 1], reverse=True)[:k]
        score = score[0]

        np.random.seed(int(score * 1000))
        for j in range(5):
            data[i + 1+k][j] = np.random.random()  # 4,5,6
        score = sorted(data[i + 4], reverse=True)[:k]
        score = score[0]

        np.random.seed(int(score * 1000))
        for j in range(5):
            data[i + 1+(2*k)][j] = np.random.random()  # 7,8,9

    for i in range(10):
        x = argmax(data[i])
        a += data[i][x]
        score = int((data[i][x]) * 10) * score

    return [argmax(s) for s in data],a


if __name__ == '__main__':
    print('beam search\n',beam())
    print('greedy search\n',greedy())

