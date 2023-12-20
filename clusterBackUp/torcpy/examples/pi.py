import time
from math import hypot
from random import random
import torcpy as torc

def test(tries):
    return sum(hypot(random(), random()) < 1 for _ in range(tries))


def calcPi(workers = 3000, tries = 15000):
    bt = time.time()
    expr = torc.map(test, [tries] * workers)
    piValue = 4. * sum(expr) / float(workers * tries)
    totalTime = time.time() - bt
    print("pi = " + str(piValue))
    print("total time: " + str(totalTime))
    return piValue


if __name__ == '__main__':
    torc.start(calcPi)