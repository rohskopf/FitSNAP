import numpy as np
import torch
import math

# Settings needed for ACE descriptors.
"""
			nradbase, #int-max k index in 
			nradmax, #int-max n index in Rnl
			lmax, #int-max l index in Rnl
			lmbda): #float - exponential factor in g(x)
"""

class ACE:
    """
    Class to calculate ACE descriptors.

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

