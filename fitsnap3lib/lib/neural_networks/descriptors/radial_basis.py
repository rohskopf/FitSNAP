import numpy as np
import torch
import math

class RadialBasis:
    """
    Class to calculate radial basis functions for a specific n, l, etc.

    Attributes:
        rc (float): Cutoff distance.
        nradbase (int): Max k index.
        nradmax (int): Max n index in Rnl.
        lmax (int): Max l index in Rnl.
        lmbda (float): Exponential factor in g(x).
    """
    def __init__(self, rc, nradbase, nradmax, lmax, lmbda):
        self.rc = rc
        self.nradbase = nradbase
        self.nradmax = nradmax
        self.lmax = lmax
        self.lmbda = lmbda

